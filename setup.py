from setuptools import setup

setup(
    name='Void_Archive',
    version='0.0.1',
    description='Void Archive is a library that oversimplified training a model and inferencing a model.',
    author='VINUK01, psw01',
    url="https://github.com/VINUK0/Void-Archive",
    include_package_data=True,
    package_data={"": ["*"]},
    install_requires=['transformers', 'peft', 'lightning', 'pandas', 'colorama'],
)
