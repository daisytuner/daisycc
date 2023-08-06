export OMP_NUM_THREADS=12

echo "DAISY (TT)"
daisycc -O3 -ffast-math -fno-unroll-loops -fopenmp -ftransfer-tune -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -I/home/lukas/anaconda3/pkgs/llvm-openmp-16.0.6-h4dfa4b3_0/include/ -o vdaisy_tt.out
for run in {1..5}; do ./vdaisy_tt.out; done

echo "DAISY (PAR)"
daisycc -O3 -ffast-math -fno-unroll-loops -fopenmp -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -I/home/lukas/anaconda3/pkgs/llvm-openmp-16.0.6-h4dfa4b3_0/include/ -o vdaisy.out
for run in {1..5}; do ./vdaisy.out; done

echo "DAISY (SEQ)"
daisycc -O3 -ffast-math -fno-unroll-loops -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -o vdaisy_seq.out
for run in {1..5}; do ./vdaisy_seq.out; done

echo "CLANG"
clang -O3 -ffast-math -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -o vclang.out
for run in {1..5}; do ./vclang.out; done

echo "POLLY"
/home/lukas/repos/llvm-project/build/bin/clang -O3 -ffast-math -mllvm -polly -mllvm -polly-parallel -mllvm -polly-tiling -mllvm -polly-vectorizer=stripmine -mllvm -polly-2nd-level-tiling -lgomp -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -o vpolly.out
for run in {1..5}; do ./vpolly.out; done

echo "ICC"
icc -O3 -ffast-math -parallel -qopenmp -DPOLYBENCH_TIME -DLARGE_DATASET atax.c ../polybench.c -I../ -o vicc.out
for run in {1..5}; do ./vicc.out; done

