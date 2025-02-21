# Notebook: Handling Imbalanced Datasets with Cluster Centroid Undersampling

## Overview

This notebook provides a hands-on demonstration of how to address imbalanced datasets using the Cluster Centroid Undersampling technique.  Class imbalance is a common problem in machine learning where one class in the target variable is significantly more frequent than the others. This can lead to biased models that perform poorly on the minority class.

This notebook walks you through the process of:

1.  **Creating a synthetic imbalanced dataset.**
2.  **Visualizing the class imbalance.**
3.  **Applying Cluster Centroid Undersampling to balance the dataset.**
4.  **Visualizing the resampled dataset to observe the effect of undersampling.**

This is a practical introduction to handling imbalanced data and provides a foundation for applying similar techniques to real-world datasets.

## Purpose and Motivation

Imbalanced datasets pose a significant challenge in machine learning because standard algorithms tend to be biased towards the majority class.  This means that while overall accuracy might seem high, the model might be very poor at identifying instances of the minority class, which is often the class of greater interest (e.g., fraud detection, disease diagnosis).

Cluster Centroid Undersampling is a technique to mitigate this issue by reducing the number of samples in the majority class. It achieves this by clustering the majority class samples and replacing each cluster with its centroid. This aims to retain the overall information of the majority class while reducing its dominance and balancing the class distribution.

This notebook is designed to:

*   **Illustrate the problem of class imbalance visually and numerically.**
*   **Introduce Cluster Centroid Undersampling as a solution.**
*   **Provide a simple, reproducible example that you can adapt and experiment with.**
*   **Show how to use the `imblearn` library in Python, a powerful tool for handling imbalanced datasets.**

## Techniques Demonstrated

This notebook showcases the following key techniques and Python libraries:

*   **Data Manipulation with `pandas`:** Creating and structuring datasets using Pandas DataFrames.
*   **Numerical Operations with `collections.Counter`:** Efficiently counting class occurrences to detect imbalance.
*   **Data Visualization with `matplotlib.pyplot` and `seaborn`:** Creating scatter plots to visualize class distributions and the effect of undersampling. Seaborn enhances visualizations for better insights.
*   **Cluster Centroid Undersampling with `imblearn.under_sampling.ClusterCentroids`:** Applying this specific undersampling technique from the `imblearn` library to balance the dataset.

## Notebook Structure - Step-by-Step Breakdown

The notebook is structured into the following steps, designed to be followed sequentially:

**1. Importing Libraries:**

   - **Code:**
     ```python
     import collections
     import pandas as pd
     import matplotlib.pyplot as plt
     import seaborn as sns
     from imblearn.under_sampling import ClusterCentroids
     ```
   - **Description:** This step imports all the necessary Python libraries that will be used throughout the notebook.
     - `collections.Counter`: For counting the occurrences of different classes in the 'ANIMAL' column.
     - `pandas`: For creating and manipulating the dataset as a DataFrame.
     - `matplotlib.pyplot`: For basic plotting functionalities, especially for scatter plots.
     - `seaborn`: For creating enhanced and visually appealing statistical plots, specifically scatter plots with class differentiation using `hue`.
     - `imblearn.under_sampling.ClusterCentroids`:  This imports the specific Cluster Centroid Undersampling class from the `imblearn` library, which is the core technique being demonstrated.
   - **Purpose:**  Ensures that all required tools are available for data handling, visualization, and undersampling.

**(Revised) 2. Creating the Dataset:**

   - **Code:**
     ```python
     Dataset = {'ANIMAL': ['DOG', 'DOG', 'DOG', 'CAT','DOG'],
                'Height': [1, 2, 2.5, 0.6,1.5], # in feet (ft)
                'Weight': [5, 10, 15, 2.5,7]} # in kilograms (kg)
     dataframe = pd.DataFrame(Dataset)
     x = dataframe[['Height', 'Weight']]
     y = dataframe['ANIMAL']
     ```
   - **Description:** This step creates a synthetic dataset based on animal characteristics, specifically Dogs and Cats, and their Height and Weight. This dataset will be used to demonstrate imbalanced data handling.  
     - A dictionary `Dataset` is defined to hold the data, with 'ANIMAL' as the target variable (values are 'DOG' and 'CAT') and 'Height', 'Weight' as features.  Units for Height are in feet (ft) and for Weight are in kilograms (kg) to align with the example dataset.
     - This dictionary is converted into a pandas DataFrame named `dataframe` for easier manipulation.
     - The 'ANIMAL' column is assigned to `y` as the dependent variable (target), and the 'Height' and 'Weight' columns are assigned to `x` as the independent variables (features).
   - **Purpose:** To establish a controlled dataset based on the animal example to demonstrate the imbalance and undersampling process. The dataset is intentionally designed to be imbalanced (more 'DOG' instances) and easy to visualize using Height and Weight features. **This version now uses only four observations as presented in the dataset image.**

**(Revised) Expected Output and Observations:**

When you run the notebook, you should expect to see the following:

*   **Output of `Counter(dataframe['ANIMAL'])`:**  This will show an imbalanced class distribution based on four observations, for example, something like `Counter({'DOG': 4, 'CAT': 1})`, indicating that 'DOG' class is more frequent than 'CAT' class, reflecting the dataset with four rows in the image.
    ... (rest of the "Expected Output and Observations" section remains largely the same, as the overall concept is still illustrated)taframe['ANIMAL']
     ```
   - **Description:** This step creates a synthetic dataset that mimics a scenario with two classes ('ANIMAL' - 0 for dog, 1 for cat) and two numerical features ('Height' and 'Weight').
     - A dictionary `Dataset` is defined to hold the data, with 'ANIMAL' as the target variable and 'Height', 'Weight' as features.
     - This dictionary is converted into a pandas DataFrame named `dataframe` for easier manipulation.
     - The 'ANIMAL' column is assigned to `y` as the dependent variable (target), and the 'Height' and 'Weight' columns are assigned to `x` as the independent variables (features).
   - **Purpose:** To establish a controlled dataset to demonstrate the imbalance and undersampling process. The dataset is intentionally designed to be imbalanced and easy to visualize.

**3. Checking for Imbalance:**

   - **Code:**
     ```python
     print(Counter(dataframe['ANIMAL']))
     ```
   - **Description:**  This line of code uses `collections.Counter` to count the occurrences of each unique value in the 'ANIMAL' column of the DataFrame.
   - **Purpose:** To quantitatively confirm the class imbalance in the dataset. The output will show the counts for each class (0 and 1), clearly indicating if one class is significantly more frequent than the other. In this example, you will observe more instances of class 0 than class 1.

**4. Visualizing Imbalanced Data:**

   - **Code:**
     ```python
     sns.scatterplot(data=dataframe, x="Height", y="Weight", hue="ANIMAL")
     plt.show()
     ```
   - **Description:** This step creates a scatter plot to visually represent the imbalanced dataset.
     - `sns.scatterplot(...)`:  Uses `seaborn.scatterplot` to generate a scatter plot.
       - `data=dataframe`: Specifies the DataFrame to use as the data source.
       - `x="Height", y="Weight"`: Sets 'Height' as the x-axis and 'Weight' as the y-axis of the plot.
       - `hue="ANIMAL"`:  Crucially, the `hue` parameter colors the data points based on the 'ANIMAL' class. This allows for visual discrimination between the two classes.
     - `plt.show()`: Displays the generated plot.
   - **Purpose:** To visually demonstrate the class imbalance. By observing the scatter plot, you will clearly see if one class (color) is more densely populated than the other, confirming the imbalance detected numerically in the previous step.

**5. Cluster Centroid Undersampling:**

   - **Code:**
     ```python
     cc = ClusterCentroids(random_state=42)
     x_resampled, y_resampled = cc.fit_resample(x, y)
     ```
   - **Description:** This step applies the Cluster Centroid Undersampling technique using `imblearn`.
     - `cc = ClusterCentroids(random_state=42)`:  Initializes the `ClusterCentroids` object. `random_state=42` ensures reproducibility of the undersampling process, meaning the same clusters and centroids will be generated each time the code is run.
     - `x_resampled, y_resampled = cc.fit_resample(x, y)`: This is the core of the undersampling operation. The `fit_resample` method:
       -  Takes the original features `x` and target variable `y` as input.
       -  Applies the Cluster Centroid Undersampling algorithm.
       -  Returns the resampled features `x_resampled` and the resampled target variable `y_resampled`.
   - **Purpose:** To balance the dataset by reducing the number of samples in the majority class using the Cluster Centroid Undersampling method.

**6. Viewing Resampled Data:**

   - **Code:**
     ```python
     print('Resampled X:\n', x_resampled)
     print('Resampled Y:\n', y_resampled)
     ```
   - **Description:** This step prints the resampled feature set (`x_resampled`) and target variable (`y_resampled`).
   - **Purpose:** To allow you to examine the numerical output of the undersampling process. By printing these variables, you can inspect the resampled data and observe how the features and class labels have been modified after applying Cluster Centroid Undersampling. You will notice a reduction in the number of samples compared to the original dataset.

**7. Visualizing Resampled Data:**

   - **Code:**
     ```python
     sns.scatterplot(x=x_resampled['Height'], y=x_resampled['Weight'], hue=y_resampled)
     plt.show()
     ```
   - **Description:**  This step generates a scatter plot of the *resampled* data.
     - `sns.scatterplot(...)`:  Creates another scatter plot using `seaborn.scatterplot`.
       - `x=x_resampled['Height'], y=x_resampled['Weight']`:  Uses the 'Height' and 'Weight' columns from the *resampled* feature set `x_resampled` for the plot axes.
       - `hue=y_resampled`:  Uses the *resampled* target variable `y_resampled` to color the data points.
     - `plt.show()`: Displays the plot of the resampled data.
   - **Purpose:** To visually assess the effect of Cluster Centroid Undersampling. By comparing this plot to the plot from Step 4, you should observe a more balanced distribution of the classes (colors). The visualization should demonstrate that the undersampling technique has helped in mitigating the initial class imbalance, leading to a more even representation of both classes in the feature space.

## How to Use This Notebook

1.  **Prerequisites:** Ensure you have the following Python libraries installed. You can install them using pip:
    ```bash
    pip install pandas matplotlib seaborn imblearn collections-counter  # collections-counter might not be needed separately as it's part of collections
    ```
    (You might only need `pip install pandas matplotlib seaborn imblearn`)

2.  **Open the Notebook:** Open this notebook (e.g., in Jupyter Notebook, JupyterLab, VS Code with Jupyter extension, or Google Colab).

3.  **Run Cells Sequentially:** Execute the notebook cells in order from top to bottom. You can do this by selecting a cell and pressing `Shift + Enter` (or the "Run" button in your notebook environment).

4.  **Observe the Output:** After running each step, carefully examine the output:
    - **Step 3:** Note the output of `Counter(dataframe['ANIMAL'])` to understand the initial class distribution.
    - **Step 4:** Observe the scatter plot. Notice the visual imbalance in the distribution of the two classes (colors).
    - **Step 6:** Review the printed `x_resampled` and `y_resampled` to see the resampled data structure.
    - **Step 7:** Examine the scatter plot of the resampled data. Compare it with the plot from Step 4. You should see a more balanced visual distribution of the two classes, indicating that the Cluster Centroid Undersampling has reduced the class imbalance.

## Expected Output and Observations

When you run the notebook, you should expect to see the following:

*   **Output of `Counter(dataframe['ANIMAL'])`:**  This will show an imbalanced class distribution, for example, something like `Counter({0: 15, 1: 5})`, indicating that class 0 is much more frequent than class 1.
*   **Scatter Plot of Imbalanced Data (Step 4):** You will see a scatter plot where one color (representing the majority class) is significantly more prevalent than the other color (representing the minority class).
*   **Printed `x_resampled` and `y_resampled` (Step 6):**  You'll see the resampled data printed. Notice that the number of rows in `x_resampled` will be less than in the original `x`, and the number of unique values in `y_resampled` will still be 0 and 1, but their distribution will be more balanced.
*   **Scatter Plot of Resampled Data (Step 7):** This scatter plot should visually demonstrate a more balanced distribution of the two classes. Compare this to the scatter plot from Step 4 to see the visual impact of Cluster Centroid Undersampling in mitigating the class imbalance.

## Key Concepts and Takeaways

By running this notebook, you should gain a better understanding of:

*   **Class Imbalance:** How to identify and visualize class imbalance in a dataset.
*   **The Impact of Imbalance:**  Understand why imbalanced datasets can be problematic for machine learning models.
*   **Cluster Centroid Undersampling:**  Learn how Cluster Centroid Undersampling works as a technique to address class imbalance by reducing the majority class samples through clustering and centroid representation.
*   **Using `imblearn`:** Get practical experience with the `imblearn` library for handling imbalanced datasets in Python.
*   **Visual Validation:** Appreciate the importance of visualizing data to understand the effects of data preprocessing techniques like undersampling.

## Further Exploration

This notebook provides a basic introduction. You can extend your learning by:

*   **Experimenting with Different Datasets:**  Try applying Cluster Centroid Undersampling to other imbalanced datasets, either synthetic or real-world datasets.
*   **Comparing with Other Undersampling Techniques:** Investigate and implement other undersampling techniques available in `imblearn`, such as Random Undersampling or NearMiss, and compare their effects.
*   **Exploring Oversampling Techniques:** Learn about oversampling methods like SMOTE (Synthetic Minority Over-sampling Technique) and compare their effectiveness with undersampling.
*   **Evaluating Model Performance:**  Train a machine learning model (e.g., Logistic Regression, Decision Tree) on both the original imbalanced dataset and the cluster centroid undersampled dataset. Compare performance metrics (accuracy, precision, recall, F1-score, AUC) to quantify the impact of undersampling on model performance, especially for the minority class.
*   **Tuning `ClusterCentroids` Parameters:**  Explore the parameters of the `ClusterCentroids` class in `imblearn`, such as `sampling_strategy` and `n_clusters`, and experiment with how they affect the undersampling process and the resulting dataset.

This notebook is a starting point for exploring the important topic of handling imbalanced datasets. By experimenting and extending upon this example, you can gain valuable practical skills for building robust and fair machine learning models in real-world scenarios.

