# AOWPD

## Building

We use cmake because it's superior:
```
mkdir build
cd build
cmake ..
# base 10 is used by default if you want to use other base do
cmake .. -DBASE=2
# instead
make
```

But for kernel (which contains GPU algorithm) use this to compile:
```
nvcc --std=c++20 kernel.cu -o kernel.exe
```
