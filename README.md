# Compiling and running cuda code

```bash
# 1 path where .cu file is locted
cd ./src

# 2 Build the code
nvcc -o cuda cuda.cu #where 'cuda'will be the name of your executable

# 3 Execute the code
./cuda.exe
```