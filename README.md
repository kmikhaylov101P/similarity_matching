# Data Matching Project

This project aims to develop and test a data matching system. The system includes functionalities for data loading, preprocessing, model building, and evaluation.
It contains 3 different vectorisation methods:"tfidf", "count", "hashing" and 3 different metrics for computing similarity: "cosine", "jaccard", "levenshtein". The input data similarity can be computed for 
the activity or category or both. The various combinations can be ensembled with a mean or voting ensemble. An automated search is also implemented with the correct category prediction being used as a proxy for 
a correct prediction. 4 metrics are implemented: accuracy, balanced accuracy, kappa and mcc. The category data can be further pre-processed by merging similar categories determined by comparing their inter and intra similarities, further remaining small categories can be combined into an "other" category. 

Further enhancements could include implementing other similarity, vectorisation and ensemble types, directly fitting to predict the categories, semi-supervised learning approaches when more uncategorised data is available and conformal effency approches if more than 1 similar entry is desired and confidence is more important. 

The availability of the C02 data and other data points that could be used as a proxy for correctness would also be highly beneficial as the applicability when just having the category is limited. Data augmentation/enhanment approaches could also be implemented to further enhance the accuracy. 

The results predicitons for 4 cases have been pre-computed and loaded onto the corresponding google drive folder: A simple case with tf-idf with cosine similarity, a mean ensemble with everything with only activity data, a mean ensemble with everything with activity and category data, and finally a ensemble identified with the best combo with a bacc metric algorithmically. 

The performance of the 4 on the test data is qualitiatively in the same ballpark. If one had to be used then I would suggest the fitted ensemble from a "sales" point of view and a reduced temptation to select the best combination of models based upon the test set data. In reality, more work would be required to have a fully satisfactory solution, in particular given that the desired prediction/model evaluation is based upon the correctness of the co2 emmisions that are not considered here. 


Note this is very much a quick hack/intial research code, further investigation/optimisation would be required prior to production.
## Project Structure

- `main.py`: Main script for executing the data matching process.
- `config.json`: Configuration file for setting various options and parameters.
- `environment.yaml`: Conda environment file for setting up the required dependencies.
- `data/`: Directory containing the input data files.

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd data_matching_clean
   ```

2. **Create and activate the conda environment:**
   ```sh
   conda env create -f environment.yaml
   conda activate activities-env
   ```

3. **Install additional dependencies (if any):**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Configure the settings:**
   Edit the `config.json` file to set the desired options and parameters.

2. **Run the main script:**
   ```sh
   python main.py
   ```

3. **View the results:**
   The output predictions will be saved to the file specified in the `config.json` (`output_file_name`).

## Configuration Options

- `merge_categories_option`: Whether to merge similar categories.
- `group_small_categories`: Whether to group small categories into an "other" category.
- `size_cutoff_for_other`: Size threshold for grouping small categories.
- `use_prespecified_config`: Whether to use a prespecified configuration for the model.
- `performance_metric`: Metric to use for evaluating the model performance.
- `activities_csv`: Path to the activities CSV file.
- `input_data_csv`: Path to the input data CSV file.
- `output_file_name`: Name of the output file for saving predictions.
- `prespecified_config`: Prespecified configuration for the model.
- `vectorizer_options`: List of vectorizers to use.
- `similarity_options`: List of similarity measures to use.
- `model_type_options`: List of model types to use.
- `ensemble_options`: List of ensemble methods to use.

## Example

An example configuration (`config.json`) is provided in the repository. You can modify it according to your needs.