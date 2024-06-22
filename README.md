# [Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs](https://arxiv.org/abs/2405.03103)
By [Jordan Dotzel](https://jordandotzel.com), [Yuzong Chen](https://yc2367.github.io/), Bahaa Kotb, Sushma Prasad, Gang Wu, Sheng Li, [Mohamed S. Abdelfattah](https://www.mohsaied.com/), [Zhiru Zhang](https://www.csl.cornell.edu/~zhiruz/index.html)


![graphical_abstract](/assets/quantized_values.png)


## Abstract

The increasing size of large language models (LLMs) traditionally requires low-precision integer formats to meet strict latency and power demands. Yet recently, alternative formats such as Normal Float (NF4) have increased model accuracy at the cost of increased chip area. In this work, we first conduct a large-scale analysis of LLM weights and activations across 30 networks and conclude that most distributions follow a Student’s t-distribution. We then derive a new theoretically optimal format, Student Float (SF4), that improves over NF4 across modern LLMs, for example increasing the average accuracy on LLaMA2-7B by 0.76% across tasks. Using this format as a high-accuracy reference, we then propose augmenting E2M1 with two variants of supernormal support for higher model accuracy. Finally, we explore the quality and efficiency frontier across 11 datatypes by evaluating their model accuracy and hardware complexity. We discover a Pareto curve composed of INT4, E2M1, and E2M1 with supernormal support, which offers a continuous tradeoff between model accuracy and chip area. For example, E2M1 with super-normal support increases the accuracy of Phi-2 by up to 2.19% with 1.22% area overhead, enabling more LLM-based applications to be run at four bits.


## Getting Started

To get started, create a conda environment with the required dependencies and activate it.

```bash
conda env create -f requirements.yaml
conda activate llm-datatypes
```

Then, use `run_quant.py` to run the quantization and evaluation on desired tasks. For example:

```bash
python run_quant.py --model facebook/opt-125m  --quantize --batch_size=64 --tasks lambada_openai --woq_bits=4 --woq_dtype=sf4_5 --woq_group_size=128 --woq_algo=RTN
```

With access to a slurm server, run the `run_quant_slurm.sh` script for batched evaluation:
```bash
slurm batch run_quant_slurm.sh
```


## Evaluation

Use run_quant.py to quantize and evaluate the model across common datasets. It includes support for weight and activation quantization, including with GPTQ[1] and SmoothQuant[2].

### Important Arguments

- `--quantize`: Enables model quantization.
- `--model`: Specifies the model to use (default: `EleutherAI/gpt-j-6b`).
- `--device`: Defines the device to use (default: `cuda:0`).
- `--seed`: Seed for sampling calibration data (default: `42`).
- `--tasks`: List of tasks for accuracy validation (default: `["lambada_openai", "hellaswag", "winogrande", "piqa", "wikitext"]`).

#### SmoothQuant Arguments
- `--sq`: Enables SmoothQuant.
- `--alpha`: SmoothQuant parameter (default: `0.5`).

#### Weight Only Quantization (WOQ) Arguments
- `--woq_enable_activation`: Enables activation quantization.
- `--woq_activation_quantile`: Clipping quantile for dynamic activation quantization (default: `1.0`).
- `--woq_algo`: Specifies the weight-only quantization algorithm (default: `RTN`, choices: `RTN`, `AWQ`, `TEQ`, `GPTQ`).
- `--woq_bits`: Number of bits for quantization (default: `8`).
- `--woq_group_size`: Group size for quantization (default: `-1`).
- `--woq_dtype`: Data type for quantization (default: `int`).

#### Supported Datatypes
Set `--woq_dtype` to select the desired datatype. The full list is provided in `neural-compressor/adapter/torch_utils/weight_only.py`, yet these are the most important 4-bit versions. They are defined as lists of floating-point values so the support can easier be extended.

- **NF4**: Normal Float (NF4) defined in QLoRA [3]

- **SF4_5**: Our proposed Student Float (SF4) format derived from the Student's t-distribution.

- **FP4_BASIC**:
  The standard E2M1 format with subnormal support

- **FP4_RANGE**:
  Our super-range E2M1 variant that provides higher accuracy especially on distributions with large spread.

- **FP4_PREC2**:
  Our super-precision E2M1 variant that leads to high accuracy across most distributions over E2M1.

- **FP4_LOG**:
  A symmetric 4-bit logarithmic format.

- **APOT4**:
  A 4-bit Additive-Powers-of-Two format.

- **APOT4_SP**:
  A 4-bit Additive-Powers-of-Two format with super-precision.

## References

1. **GPTQ**: \
   Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. \*ICLR 2023\*. [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)

2. **SmoothQuant**: \
   Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. \*ICML 2024\*. [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)

3. **QLoRA**: \
   Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. QLoRA: Efficient Finetuning of Quantized LLMs. \*NeurIPS 2023\*. [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)


## Acknowledgements

This code was built from the Intel Neural Compressor [codebase](https://github.com/intel/neural-compressor).

## Citation
```
@article{dotzel2024students,
      title={Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs}, 
      author={Jordan Dotzel and Yuzong Chen and Bahaa Kotb and Sushma Prasad and Gang Wu and Sheng Li and Mohamed S. Abdelfattah and Zhiru Zhang},
      year={2024},
      journal={International Conference on Machine Learning}
}
```