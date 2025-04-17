from setuptools import setup,find_packages


def get_requirements(file_path:str)->list:
    """This function returns a list of requirements."""

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
        
    return requirements



#This is a setup file for the MLproject package.
setup(
    name="MLproject",
    version="0.0.1",
    author="santhankumar",
    author_emaail="kumarsanthan32@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirement.txt"),
    description="A small ML Project",
)