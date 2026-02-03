# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil

from setuptools import setup

# Get the directory where this setup.py file is located
setup_dir = os.path.dirname(os.path.abspath(__file__))

# make the faiss python package dir
shutil.rmtree("faiss", ignore_errors=True)
os.mkdir("faiss")

# Copy files using absolute paths based on setup.py location
if os.path.exists(os.path.join(setup_dir, "contrib")):
    shutil.copytree(os.path.join(setup_dir, "contrib"), "faiss/contrib")

shutil.copyfile(os.path.join(setup_dir, "__init__.py"), "faiss/__init__.py")
shutil.copyfile(os.path.join(setup_dir, "loader.py"), "faiss/loader.py")
shutil.copyfile(os.path.join(setup_dir, "class_wrappers.py"), "faiss/class_wrappers.py")
shutil.copyfile(os.path.join(setup_dir, "gpu_wrappers.py"), "faiss/gpu_wrappers.py")
shutil.copyfile(os.path.join(setup_dir, "extra_wrappers.py"), "faiss/extra_wrappers.py")
shutil.copyfile(os.path.join(setup_dir, "array_conversions.py"), "faiss/array_conversions.py")
gpu_pool_controller_src = os.path.join(setup_dir, "gpu_pool_controller.py")
if os.path.exists(gpu_pool_controller_src):
    shutil.copyfile(gpu_pool_controller_src, "faiss/gpu_pool_controller.py")

# Copy utils.py only if it exists
utils_src = os.path.join(setup_dir, "utils.py")
if os.path.exists(utils_src):
    shutil.copyfile(utils_src, "faiss/utils.py")

has_engine = os.path.exists(os.path.join(setup_dir, "engine"))
has_server = os.path.exists(os.path.join(setup_dir, "server"))
if has_engine:
    shutil.copytree(os.path.join(setup_dir, "engine"), "faiss/engine")
if has_server:
    shutil.copytree(os.path.join(setup_dir, "server"), "faiss/server")

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"
prefix = "Release/" * (platform.system() == "Windows")

swigfaiss_generic_lib = f"{prefix}_swigfaiss{ext}"
swigfaiss_avx2_lib = f"{prefix}_swigfaiss_avx2{ext}"
swigfaiss_avx512_lib = f"{prefix}_swigfaiss_avx512{ext}"
swigfaiss_avx512_spr_lib = f"{prefix}_swigfaiss_avx512_spr{ext}"
callbacks_lib = f"{prefix}libfaiss_python_callbacks{ext}"
swigfaiss_sve_lib = f"{prefix}_swigfaiss_sve{ext}"
faiss_example_external_module_lib = f"_faiss_example_external_module{ext}"

found_swigfaiss_generic = os.path.exists(os.path.join(setup_dir, swigfaiss_generic_lib))
found_swigfaiss_avx2 = os.path.exists(os.path.join(setup_dir, swigfaiss_avx2_lib))
found_swigfaiss_avx512 = os.path.exists(os.path.join(setup_dir, swigfaiss_avx512_lib))
found_swigfaiss_avx512_spr = os.path.exists(os.path.join(setup_dir, swigfaiss_avx512_spr_lib))
found_callbacks = os.path.exists(os.path.join(setup_dir, callbacks_lib))
found_swigfaiss_sve = os.path.exists(os.path.join(setup_dir, swigfaiss_sve_lib))
found_faiss_example_external_module_lib = os.path.exists(
    os.path.join(setup_dir, faiss_example_external_module_lib)
)

if platform.system() != "AIX":
    assert (
        found_swigfaiss_generic
        or found_swigfaiss_avx2
        or found_swigfaiss_avx512
        or found_swigfaiss_avx512_spr
        or found_swigfaiss_sve
        or found_faiss_example_external_module_lib
    ), (
        f"Could not find {swigfaiss_generic_lib} or "
        f"{swigfaiss_avx2_lib} or {swigfaiss_avx512_lib} or {swigfaiss_avx512_spr_lib} or {swigfaiss_sve_lib} or {faiss_example_external_module_lib}. "
        f"Faiss may not be compiled yet."
    )

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_lib}")
    swigfaiss_py = os.path.join(setup_dir, "swigfaiss.py")
    if os.path.exists(swigfaiss_py):
        shutil.copyfile(swigfaiss_py, "faiss/swigfaiss.py")
    swigfaiss_lib_path = os.path.join(setup_dir, swigfaiss_generic_lib)
    if os.path.exists(swigfaiss_lib_path):
        shutil.copyfile(swigfaiss_lib_path, f"faiss/_swigfaiss{ext}")

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_lib}")
    swigfaiss_avx2_py = os.path.join(setup_dir, "swigfaiss_avx2.py")
    if os.path.exists(swigfaiss_avx2_py):
        shutil.copyfile(swigfaiss_avx2_py, "faiss/swigfaiss_avx2.py")
    swigfaiss_avx2_lib_path = os.path.join(setup_dir, swigfaiss_avx2_lib)
    if os.path.exists(swigfaiss_avx2_lib_path):
        shutil.copyfile(swigfaiss_avx2_lib_path, f"faiss/_swigfaiss_avx2{ext}")

if found_swigfaiss_avx512:
    print(f"Copying {swigfaiss_avx512_lib}")
    swigfaiss_avx512_py = os.path.join(setup_dir, "swigfaiss_avx512.py")
    if os.path.exists(swigfaiss_avx512_py):
        shutil.copyfile(swigfaiss_avx512_py, "faiss/swigfaiss_avx512.py")
    swigfaiss_avx512_lib_path = os.path.join(setup_dir, swigfaiss_avx512_lib)
    if os.path.exists(swigfaiss_avx512_lib_path):
        shutil.copyfile(swigfaiss_avx512_lib_path, f"faiss/_swigfaiss_avx512{ext}")

if found_swigfaiss_avx512_spr:
    print(f"Copying {swigfaiss_avx512_spr_lib}")
    swigfaiss_avx512_spr_py = os.path.join(setup_dir, "swigfaiss_avx512_spr.py")
    if os.path.exists(swigfaiss_avx512_spr_py):
        shutil.copyfile(swigfaiss_avx512_spr_py, "faiss/swigfaiss_avx512_spr.py")
    swigfaiss_avx512_spr_lib_path = os.path.join(setup_dir, swigfaiss_avx512_spr_lib)
    if os.path.exists(swigfaiss_avx512_spr_lib_path):
        shutil.copyfile(swigfaiss_avx512_spr_lib_path, f"faiss/_swigfaiss_avx512_spr{ext}")

if found_callbacks:
    print(f"Copying {callbacks_lib}")
    callbacks_lib_path = os.path.join(setup_dir, callbacks_lib)
    if os.path.exists(callbacks_lib_path):
        shutil.copyfile(callbacks_lib_path, f"faiss/{callbacks_lib}")

if found_swigfaiss_sve:
    print(f"Copying {swigfaiss_sve_lib}")
    swigfaiss_sve_py = os.path.join(setup_dir, "swigfaiss_sve.py")
    if os.path.exists(swigfaiss_sve_py):
        shutil.copyfile(swigfaiss_sve_py, "faiss/swigfaiss_sve.py")
    swigfaiss_sve_lib_path = os.path.join(setup_dir, swigfaiss_sve_lib)
    if os.path.exists(swigfaiss_sve_lib_path):
        shutil.copyfile(swigfaiss_sve_lib_path, f"faiss/_swigfaiss_sve{ext}")

if found_faiss_example_external_module_lib:
    print(f"Copying {faiss_example_external_module_lib}")
    faiss_example_py = os.path.join(setup_dir, "faiss_example_external_module.py")
    if os.path.exists(faiss_example_py):
        shutil.copyfile(faiss_example_py, "faiss/faiss_example_external_module.py")
    faiss_example_lib_path = os.path.join(setup_dir, faiss_example_external_module_lib)
    if os.path.exists(faiss_example_lib_path):
        shutil.copyfile(
            faiss_example_lib_path,
            f"faiss/_faiss_example_external_module{ext}",
        )

long_description = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
packages_list = ["faiss"]
if os.path.exists(os.path.join(setup_dir, "contrib")):
    packages_list.extend(["faiss.contrib", "faiss.contrib.torch"])
if has_engine:
    packages_list.append("faiss.engine")
if has_server:
    packages_list.append("faiss.server")

setup(
    name="faiss",
    version="1.13.2",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging"],
    packages=packages_list,
    package_data={
        "faiss": ["*.so", "*.pyd", "*.a"],
    },
    zip_safe=False,
)
