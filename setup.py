from setuptools import setup, find_packages

setup(
    name='rl',
    version='0.1.0',
    description='RL for internal use',
    author='Neev Parikh',
    author_email=('neev_parikh@brown.edu'),
    packages=find_packages(include=['rl', 'rl.*']),
    install_requires=[
        "torch",
        "torchvision",
        "tensorboard",
        "numpy",
        "gym[all]",
        "atariari @ git+ssh://git@github.com/mila-iqia/atari-representation-learning@master#egg=atariari",
        "seaborn",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ])
