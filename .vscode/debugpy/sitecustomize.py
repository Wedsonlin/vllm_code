import subprocess

# Python 3.11 always uses vfork for subprocesses. Under debugpy that
# exec fails with EACCES for non-Python programs such as ninja, which
# FlashInfer runs while compiling sampling kernels.
subprocess._USE_VFORK = False
