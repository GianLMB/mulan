from setuptools import setup


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line]  # and not line.startswith('#')]
    

with open("README.md", "r") as f:
    readme = f.read()
    
sources = {
    "mulan": "mulan",
    "mulan.scripts": "scripts",
}

setup(
    name='mulan',
    version='0.1.0',
    description="MuLAN: MUtational effects with Light Attention Networks",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Gianluca Lombardi",
    author_email="gianluca.lombardi@sorbonne-universite.fr",
    url="https://github.com/GianLMB/mulan",
    license="CC BY-NC-SA 4.0",
    packages=sources.keys(),
    package_dir=sources,
    python_requires='>=3.9',
    install_requires= parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'plm-embed=mulan.scripts.generate_embeddings:main',
            'mulan-predict=mulan.scripts.predict:main',
            'mulan-att=mulan.scripts.extract_attentions:main',
            'mulan-landscape=mulan.scripts.compute_landscape:main',
            'mulan-train=mulan.scripts.train:main',
        ],
    },
)
