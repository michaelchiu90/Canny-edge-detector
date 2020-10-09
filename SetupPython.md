# Setup Python Environment
You need Python3 and several Python packages, including Numpy, Pillow, and PyTorch, to finish this course. If you know well about how to setup and manage Python environment, you may install python and these packages in your preferred way and ignore this section. If not, here we recommend you use Anaconda and PyCharm. This section helps you setup a Python3 development environment step by step.

## Install Anaconda
The most convenient way to setup a Python environment is to install Anaconda, a Python environment management system with lots of useful third-party packages. Anaconda already contains nearly all packages we need.

Please go to [Anaconda download page](https://www.anaconda.com/products/individual), click "Download" and install correct one for your operating system.

<!-- ### Validate your Install
After install, open "Anaconda Prompt" from your Windows Start menu (or corresponding terminal in OSX and Linux), and type command `conda --version` and `python --version`. If you see following output, you have installed Conda successfully.

```bash
(base) C:\>conda --version
conda 4.8.2

(base) C:\>python --version
Python 3.7.4
``` -->

<!-- ## Install IPython
IPython is the most popular interactive shell for python and will be the major teaching tools in our tutorials. Beginners can try anything they just learn in IPython quickly. You could install ipython via following command in Anaconda Prompt.

```bash
(base) C:\>conda install ipython
...
Proceed ([y]/n)?y
...
``` -->

## Install PyCharm
If you need an integrated development environment (IDE), I recommend PyCharm. PyCharm is the most popular Python IDE in the world. It helps you manage, run, debug, and deploy Python programs. Now, a free [community version](https://www.jetbrains.com/pycharm/download/) is available. You can download and install it.

After install PyCharm, you need to configure it to use Anaconda you installed before.

1. Open PyCharm configuration by pressing
    - "__Ctrl__ + __Alt__ + __s__" on Windows, or
    - "__Cmd⌘__ + __,__" on macOS.
2. Search "interpreter" in left up corner.
3. Choose "⚙ -> __Add__" at right side.
4. In the new interface, choose "__Conda Environment__ -> __Existing environment__"
5. Select "__Interpreter__" and "__Conda executable__" as the one you installed:
    - Example on Win10:
        - Interpreter:  `C:\Users\<USERNAME>\anaconda3\python.exe`
        - Conda executable: `C:\Users\<USERNAME>\anaconda3\Scripts\conda.exe`
    - Example on macOS:
        - Interpreter: `/Users/<USERNAME>/anaconda3/bin/python`
        - Conda executable: no need to select.
6. Check "__Make available to all projects__" and click "__OK__".
7. Now you come back to previous interface. Wait for it finishing processing, then click "__OK__" again.

After configuring, PyCharm projects now use Anaconda as default Python environment.

## Official Python Tutorial
This course presumes you all have learned Python before. If not, please read the official [Python Tutorial](https://docs.python.org/3/tutorial/index.html), which is one of the best python tutorials in the world. Even if you have learned python before, I also recommend you go through the whole official tutorial quickly.

#### How to get help information in Python
You may forget usage of a function sometimes. In this case, you can use built in function `help` to get help information about the function. E.g., if you forget how to use `print` function, you can get help like this:
```python
help(print)
```
You will see a help information like this, which tells you the meaning of each argument:
```
Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.

    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.
```
