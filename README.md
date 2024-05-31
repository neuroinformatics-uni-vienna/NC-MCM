# NC-MCM 
<a href=https://github.com/DriftKing1998/NC-MCM-Visualizer>See on GitHub</a>
## A toolbox to visualize neuronal imaging data and apply the NC-MCM framework to it

This is a toolbox uses neuronal & behavioral data and visualizes it. The main functionalities include: 
* `N`umerous options to visualize neuronal data and create diagnostic plots 
* `C`reating neural manifolds using `sklearn` dimensionality reduction algorithms or `BunDLeNet`
* `M`ake interactive behavioral state diagrams using `pyvis`
* `C`luster behavioral probability trajectories and test them for non-markovianity
* `M`ovies of behavioral/neuronal trajectories saved as `.gif`-files

### These are some of the plots created from calcium imaging data of C. elegans
#### Interactive behavioral state diagram for worm 3 and 3 cognitive states (saved as a .html file)
<img src="_OLD_ncmcm/Demonstration/InteractivePlot.png" width="700" alt="Behavioral State Diagram for Worm 3 and 3 cognitive states - interactive">

#### Comparison of predicted and true label using BunDLeNet's tau model as mapping and its predictor on worm 3
<img src="_OLD_ncmcm/Demonstration/ComaprisonBunDLeNet.png" width="700" alt="Comparison between true and predicted label using BunDLeNet as mapping and predictor">

#### Movie using BunDLeNet's tau model as mapping on worm 1
<img src="_OLD_ncmcm/Demonstration/Worm_1_Interval_100.gif" width="700" alt="Movie using BunDLeNet's tau model as mapping and the true labels">

## Getting Started (for end-users)
1. **Installation:** Open a terminal window in your Python project directory and run
    ```
    pip install ncmcm
    ```
2. **Importing the package:** In your Python script or notebook, import the package
    ```
    import ncmcm
    ```
3. **Usage:** the `ncmcm.Database` class as a container for your neuronal and behavioral dataset
4. **Tutorial:** Check out the `Demo.ipynb` notebook included in the package. It serves as a useful starting point to explore the functionalities of `ncmcm`

## Installation and usage information (for contributors)

If you're interested in contributing to this project or creating your own versions based on the existing code, follow these steps:

### Installation

1. **Clone the Repository**: 
   Clone this repository to your local machine using Git:
   ```
   git clone https://github.com/DriftKing1998/NC-MCM-Visualizer.git
   ```

2. **Install Dependencies**: 
   Navigate to the project directory and install the required dependencies using pip:
   ```
   cd <project_directory>
   pip install -r requirements.txt
   ```

### New Branches

1. **Explore the Code**:
    <br>`ncmcm.classes.py` contains the classes used by ncmcm:
   1. `Database` is a container for data, which can be used to generate the behavioral probability maps by adding a sklearn-model. It also allows to create different plots and diagnostics.
   2. `Visualizer` is created by adding a mapping or a BundDLeNet (=default). It allows to create 3D plots and movies.
   3. `CustomEnsembleModel` is a model that creates an ensemble of models, specializing each model to detect a label.

    <br>`ncmcm.functions.py` contains some auxiliary functions:
   1. Functions to prepare data 
   2. Functions to test for stationarity & non-markovianity
   3. Plotting functions for the tests

    <br>`ncmcm.BundDLeNet.py` contains all the parts of BundDLeNet:
   1. Class for creating/training the model 
   2. Functions to prep data and get loss
    
2. **Make Changes**:
   Make your desired changes or enhancements to the codebase. Feel free to add new features, fix bugs, or improve documentation.

3. **Testing**:
   Ensure that your changes are tested thoroughly. You can run existing tests or write new ones to validate your modifications.

4. **Create a Branch**:
   Create a new branch for your changes:
   ```
   git checkout -b <new_branch>
   ```

5. **Commit Your Changes**:
   Once you're satisfied with your changes, commit them to your branch:
   ```
   git add .
   git commit -m "description of changes"
   ```

6. **Push Changes**:
   Push your changes to your forked repository:
   ```
   git push origin <new_branch>
   ```
### Add to the Codebase
Since this project is a first step, any additions are more than welcome. 
1. **Pull request**:
   Go to the repository and open a pull request from your branch. Provide a clear description of your changes and why they're beneficial.

2. **Review**:
   Await feedback from me and address any requested changes. Once approved, your changes will be merged into the main branch.

###### This project was created as part of the masters project of *Hofer Michael* 
