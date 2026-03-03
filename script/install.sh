make -C build -j faiss
make -C build -j faiss_avx512
make -C build -j swigfaiss

cd build/faiss/python/
uv pip install --force-reinstall .

cd ../../..

