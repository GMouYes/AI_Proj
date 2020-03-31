# KalmanFilters

Our Kalman filter implementation and the associated "monthly reports" are included in the file "Part 1.xlsx".
Three workbooks are included, described below:

## 1. "**Filtering+Prediction**"

This is the bulk of our impelementation, and is largely based on the "kalman filter-1.xlsx" file from class. The first few columns start
with the initial homeless population and variance, and define the transition and sensor models used for the "Predict" and "Update" phases
of the filter. The filtering continues down the rows, generating population and variance estimates per month, until **row 223**, at which
point prediction is used to estimate the final two months (September and October).

Also included are two plots: one plot containing the monthly probability distributions for the true homeless population, along with the
transition model, sensor 1, sensor 2, and the final estimates; and one plotting monthly estimates from filtering + prediction versus the
true homeless population.

We also include a table starting at cell **J41**, which summarizes all the filter estimates and parameters for each month (this can be used
as reference for each "monthly report").

**Note:** The plot of probability distributions for the filter components has a dropdown menu to toggle the data for each month, but by 
default Excel will recalculate almost the entire spreadsheet, since a large portion depends on randomly picking values from normal 
distributions. This can be useful for seeing a large array of possible scenarios, but is not particularly useful for this dropdown menu, 
using it can alter the data you want to observe! Therefore, we recommend disabling Excel's automatic workbook calculation when interacting 
with it. In Excel 2016, this option can be found under **File -> Options -> Formulas -> Calculation options**; changing **Workbook 
Calculation** to **Manual** will make the dropdown usable.

## 2. "**Smoothing**"

Here we implement Rauch-Tung-Striebel (RTS) smoothing to revise our population estimates after our six months of data collection have 
elapsed. Both data from the forward filtering and backward smoothing processes are included, as well as a plot comparing the filtered 
estimates, smoothed estimates, and "ground truth" data.

## 3. "**Worcester City Population Data**"

Here we present the datasets and associated calculations used to inform the initial parameter values for our filter.


For more information about our filtering implementation and parameter selection, please refer to the section for **Part 1** in our project writeup.
