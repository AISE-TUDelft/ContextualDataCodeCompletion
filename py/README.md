# Important Note

The scripts in the subdirectories here are *modified* finetuning and prediction scripts.
The main modification is that they were made compatible with Weights and Biases ([https://wandb.ai](https://wandb.ai)).
If you opt to also use Weights and Biases, make sure to input your project and entity inside the `finetune_predict.py` files.
The relevant variable names are `wandb_project` and `wandb_entity`.

The original files can be found here:
- [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/code-completion)
- [CodeGPT](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line/)
- [InCoder](https://github.com/dpfried/incoder)


## CodeGPT Files
Note that CodeGPT uses identical files to UniXcoder, so you can either make CodeGPT use those files, or simply copy them for use by CodeGPT.