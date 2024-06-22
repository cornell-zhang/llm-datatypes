import argparse
import sys

sys.path.insert(0, './')

import time
import json
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from neural_compressor.adaptor.torch_utils.model_wrapper import LookupLinear, SQLinearWrapper
from neural_compressor.adaptor.torch_utils.smooth_quant import set_module
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from neural_compressor.adaptor.torch_utils.residual_utils import make_new_opt_forward


parser = argparse.ArgumentParser()
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument(
    '--seed',
    type=int, default=42, help='Seed for sampling the calibration data.'
)
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=512, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai",
                                                   "hellaswag", "winogrande", "piqa", "wikitext"],
                    type=str, help="tasks list for accuracy validation, text-generation and code-generation tasks are different.")
# ============SmoothQuant===============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default=0.5, help="Smooth quant parameter.")
# ============WeightOnly configs===============
parser.add_argument("--woq_enable_activation", action="store_true")
parser.add_argument('--woq_activation_quantile', type=float, default=1.0,
                    help='Clipping quantile for dynamic activation quantization.')
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
                    help="Weight-only parameter.")
parser.add_argument("--woq_bits", type=int, default=8)
parser.add_argument("--woq_dtype", type=str, default='int')
parser.add_argument("--woq_group_size", type=int, default=-1)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--woq_enable_full_range", action="store_true")
# =============GPTQ configs====================
parser.add_argument("--gptq_actorder", action="store_true",
                    help="Whether to apply the activation order GPTQ heuristic.")
parser.add_argument('--gptq_percdamp', type=float, default=.01,
                    help='Percent of the average Hessian diagonal to use for dampening.')
parser.add_argument('--gptq_block_size', type=int, default=128, help='Block size. sub weight matrix size to run GPTQ.')
parser.add_argument('--gptq_nsamples', type=int, default=128, help='Number of calibration data samples.')
parser.add_argument('--gptq_use_max_length', action="store_true",
                    help='Set all sequence length to be same length of args.gptq_pad_max_length')
parser.add_argument('--gptq_pad_max_length', type=int, default=2048, help='Calibration dataset sequence max length, \
                                                                           this should align with your model config, \
                                                                           and your dataset builder args: args.pad_max_length')
parser.add_argument("--residual_ratio", default=-1.0, type=float)

args = parser.parse_args()

class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196, is_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.is_calib = is_calib

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        if args.woq_algo in ['TEQ']:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            example = self.tokenizer(examples["text"], padding="max_length", max_length=self.pad_max)
        else:
            example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):

        input_ids_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            if self.is_calib:
                if args.woq_algo != 'GPTQ':
                    input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
            else:
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)

        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            outputs = model(input_ids)
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (i + 1) % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        print("Latency: ", latency)
        return acc


def get_user_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torchscript = False
    if args.sq or args.woq_algo in ['AWQ', 'TEQ']:
        torchscript = True
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=torchscript,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    user_model = user_model.float()

    # Set model's seq_len when GPTQ calibration is enabled.
    if args.woq_algo == 'GPTQ':
        user_model.seqlen = args.gptq_pad_max_length

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    return user_model, tokenizer

def extract_layers_to_scales_mapping(sq_model):
    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}

    sq_conf = PostTrainingQuantConfig(
        backend="default",
        approach="static",
        recipes=recipes,
    )

    q_sq_model = quantization.fit(
        sq_model,
        sq_conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
        eval_func=eval_func,
    )
    # Remove the '_model.' prefix from the layer names
    layers_to_scales = {name[7:]: module.input_scale for name, module in q_sq_model.named_modules() if isinstance(module, SQLinearWrapper)}
    
    # Delete the q_sq_model and sq_model to free up memory
    del q_sq_model, sq_model

    return layers_to_scales

if args.quantize:
    user_model, tokenizer = get_user_model()
    calib_dataset = load_dataset(args.dataset, split="train")
    calib_dataset = calib_dataset.shuffle(seed=args.seed)
    # Truncate the dataset to the first 1000 samples because currently fails on Llama tokenizer
    calib_dataset = calib_dataset.select(range(1000))
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )


    def calib_func(prepared_model):
        for i, calib_input in enumerate(calib_dataloader):
            if i > args.calib_iters:
                break
            prepared_model(calib_input[0])

    recipes = {}
    eval_func = None
    from neural_compressor import PostTrainingQuantConfig, quantization

    layers_to_scales = None
    if args.sq:
        # Duplicate the model to avoid quantizing the original model, only used for scales
        sq_model, _ = get_user_model()
        layers_to_scales = extract_layers_to_scales_mapping(sq_model)

        # Scale all the layers by the extracted scales
        for name, module in user_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in layers_to_scales:
                    scale = layers_to_scales[name]
                    module.weight.data = torch.div(module.weight.data, scale)

    op_type_dict = {
        '.*': {  # re.match
            "weight": {
                'dtype' : args.woq_dtype,
                'bits': args.woq_bits,  # 1-8 bits
                'group_size': args.woq_group_size,  # -1 (per-channel)
                'scheme': args.woq_scheme,  # sym/asym
                'algorithm': args.woq_algo,  # RTN/AWQ/TEQ
            },
        },
    }
    op_name_dict = {
        'lm_head': {"weight": {'dtype': 'fp32'}, },
        'embed_out': {"weight": {'dtype': 'fp32'}, },  # for dolly_v2
    }
    recipes["rtn_args"] = {
        "enable_mse_search": args.woq_enable_mse_search,
        "enable_full_range": args.woq_enable_full_range,
    }
    recipes['gptq_args'] = {
        'percdamp': args.gptq_percdamp,
        'act_order': args.gptq_actorder,
        'block_size': args.gptq_block_size,
        'nsamples': args.gptq_nsamples,
        'use_max_length': args.gptq_use_max_length,
        'pad_max_length': args.gptq_pad_max_length
    }
    # GPTQ: use assistive functions to modify calib_dataloader and calib_func
    # TEQ: set calib_func=None, use default training func as calib_func
    if args.woq_algo in ["GPTQ", "TEQ"]:
        calib_func = None

    conf = PostTrainingQuantConfig(
        approach='weight_only',
        op_type_dict=op_type_dict,
        op_name_dict=op_name_dict,
        recipes=recipes,
    )

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
        eval_func=eval_func,
    )

    # NOTE: Too complex to modify the actual traverse code and current W+A quant uses PyTorch models
    # Slightly strange to call it `woq` with activation quantization but it refers to the modules
    if args.woq_enable_activation:
        for name, module in q_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if args.sq:
                    # [7:] to remove the '_model.' prefix
                    short_name = name[7:]
                    sq_scales = layers_to_scales[short_name] if short_name in layers_to_scales else None
                else:
                    sq_scales = None
                wrapper_module = LookupLinear(orig_layer=module, num_bits=args.woq_bits, dtype=args.woq_dtype,
                                            group_size=args.woq_group_size, scheme=args.woq_scheme,
                                            sq_scales=sq_scales, quantile=args.woq_activation_quantile)
                set_module(q_model, name, wrapper_module)
    user_model = q_model
else:
    user_model, _ = get_user_model()

if args.residual_ratio > 0:
    # Override the original forward method with one that masks on the residual ratio
    new_opt_forward = make_new_opt_forward(args.residual_ratio)
    OPTDecoderLayer.forward = new_opt_forward
user_model = user_model.to(args.device)
user_model.eval()
results = evaluate(
    model="hf-causal",
    model_args='pretrained=' + args.model + ',tokenizer=' + args.model + ',dtype=float32',
    user_model=user_model,
    batch_size=args.batch_size,
    tasks=args.tasks,
    device=args.device
)

dumped = json.dumps(results, indent=2)
if args.save_accuracy_path:
    with open(args.save_accuracy_path, "w") as f:
        f.write(dumped)
for task_name in args.tasks:
    if task_name == "wikitext":
        acc = results["results"][task_name]["word_perplexity"]
    else:
        acc = results["results"][task_name]["acc"]
print("Accuracy: %.5f" % acc)
print('Batch size = %d' % args.batch_size)
