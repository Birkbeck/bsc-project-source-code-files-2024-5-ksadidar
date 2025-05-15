finalYearProject
BSC DATA SCIENCE FINAL YEAR PROJECT . ├── meta_analysis/ # Meta-analysis code and results │ ├── data/ # Extracted data from studies │ ├── analysis/ # Meta-analysis implementation │ └── results/ # Visualizations and statistical outcomes ├── recommendation_system/ # ML recommendation system implementation │ ├── data/ # Dataset for the recommendation system │ ├── models/ # Implementation of different ML algorithms │ ├── evaluation/ # Code for evaluating recommendation performance │ └── utils/ # Helper functions ├── notebooks/ # Jupyter notebooks with experimental results ├── docs/ # Project documentation └── scripts/ # Utility scripts for data processing

Meta-Analysis Approach The meta-analysis component examines published research on recommendation systems to identify:

Most effective ML algorithms for different recommendation scenarios Optimal feature selection techniques Best evaluation metrics for performance assessment Common challenges and their solutions

Python packages used for meta-analysis:

PyMeta3 statsmodels pandas matplotlib/seaborn

Recommendation System Implementation Based on the meta-analysis findings, the project implements a recommendation system with:

Multiple algorithm options (collaborative filtering, content-based, hybrid approaches) Configurable feature selection Comprehensive evaluation metrics Solutions for common challenges (cold start, data sparsity)

Technologies Used

Python: Primary programming language Pandas/NumPy: Data manipulation and numerical operations Scikit-learn: ML algorithm implementation PyTorch/TensorFlow: For deep learning models (if applicable) PyMeta3/statsmodels: Meta-analysis calculations Matplotlib/Seaborn: Visualization SQL: Database operations for storing and retrieving data

Prerequisites python >= 3.8 pandas numpy scikit-learn matplotlib seaborn pymeta3 statsmodels

License This project is licensed under the APACHE License - see the LICENSE file for details.

Acknowledgments Dr. Dimitrios Airantzis for project guidance
