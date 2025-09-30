# python setup.py sdist bdist_wheel
# twine upload dist/*

from setuptools import setup, find_packages
import logging 

logger = logging.getLogger(__name__)

def read_readme():
    with open('README.md') as f:
        return f.read()
    
def setup_flashrl_env():
    import site
    import os
    path = site.getsitepackages()[0]
    need_usercustomize = True
    if os.path.exists(os.path.join(path, 'usercustomize.py')):
        with open(os.path.join(path, 'usercustomize.py'), 'r') as f:
            for line in f.readlines():
                if 'import flash_rl' in line and not line.strip().startswith("#"):
                    logger.info("flash_rl already imported in usercustomize.py")
                    need_usercustomize = False
                    break 
                           
    if need_usercustomize:
        with open(os.path.join(path, 'usercustomize.py'), 'a') as f:
            f.write(f"try: import flash_rl\nexcept ImportError: pass\n")
            logger.info("flash_rl setup added to usercustomize.py")

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'torch',
    'transformers',
    'huggingface_hub',
]

setup_flashrl_env()

setup(
    name='flash_llm_rl',
    version='1.0.3',
    description='flash llm rl',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='https://github.com/yaof20/Flash-RL',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='License :: OSI Approved :: MIT License',
    entry_points={
        'console_scripts': ['flashrl=flash_rl.commands:run'],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
    ]
)