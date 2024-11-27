# Invert Map
This repo contains different algo of map inversion, most of them are found in this stackoverflow question: [Inverting a real-valued index grid](https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/65566295?answertab=scoredesc). 

Each algo has its own file (barycentric.py, iterative.py etc...).


## Install dependencies
The python requirements are maintained with `uv` from [astral](https://docs.astral.sh/uv/).   
```bash
uv sync
```
This single command will install python, create a venv and install the requirements. Alternatively you can use pip with the `requirements.txt`

## Run the demo
![demo.png](doc%2Fimages%2Fdemo.png)
```bash
python main_demo.py
```

The demo generates a testing image and a distortion mapping. It applies the mapping to the image. Then it runs the 

## Run the benchmark
```bash
python main_benchmark.py
```
The benchmark runs multiple algo for different image sizes and different kind of mapping.

It displays 2 plots for each mapping. In the example below we see that "iterative" is the fastest, while "barycentric2" has the lowest error
  
![CPU_time.png](doc%2Fimages%2FCPU_time.png)

![RMSE.png](doc%2Fimages%2FRMSE.png)

## Contribute
Contribution are welcome. Send your pull requests and I will merge them within a month. When submitting a new algo, please place it in a new file. Your file must contain a function with the signature below
```python
def invert_map(xmap: NDArray, ymap: NDArray) -> tuple[NDArray, NDArray]:
```
If your algo needs new requirements, please add them using [uv](https://docs.astral.sh/uv/)
```bash
uv add [package-name]
```

