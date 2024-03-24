#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation - GRIP'2024 - Data Science & Business Analytics
# ## Task : Exploratory Data Analysis - Retail SampleSuperstore
# ### Presented By : kanika Arya
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load the dataset
data = pd.read_csv("SampleSuperstore.csv")

data.head()


# In[3]:


data.info()


# In[4]:


# Summary statistics
print(data.describe())


# In[5]:


# Check for missing values
print(data.isnull().sum())


# In[6]:


# Calculate total sales and total profit
total_sales = data['Sales'].sum()
total_profit = data['Profit'].sum()
print(f"Total Sales: {total_sales}, Total Profit: {total_profit}")


# In[7]:


# Sales and profit by category
category_analysis = data.groupby('Category').agg(Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum'))
print(category_analysis)


# In[8]:


# Sales and profit by sub-category
subcategory_analysis = data.groupby('Sub-Category').agg(Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum'))
print(subcategory_analysis.sort_values(by='Total_Profit'))


# In[9]:


# Analyzing the impact of discounts on profitability
discount_impact = data.groupby(['Category', 'Sub-Category']).agg(Average_Discount=('Discount', 'mean'),
                                                                  Total_Sales=('Sales', 'sum'),
                                                                  Total_Profit=('Profit', 'sum'))
print(discount_impact.sort_values(by=['Category', 'Total_Profit']))


# In[10]:


# Geographical analysis by state
state_analysis = data.groupby('State').agg(Total_Sales=('Sales', 'sum'),
                                           Total_Profit=('Profit', 'sum')).sort_values(by='Total_Profit')
print(state_analysis.head(10))


# In[11]:


# Customer segment analysis
segment_analysis = data.groupby('Segment').agg(Total_Sales=('Sales', 'sum'),
                                                Total_Profit=('Profit', 'sum')).sort_values(by='Total_Profit')
print(segment_analysis)


# In[12]:


# Sales by category
sns.barplot(x='Category', y='Total_Sales', data=category_analysis.reset_index())
plt.title('Sales by Category')
plt.show()


# In[13]:


# Profit by state (top 10 negative profits)
top_10_states = state_analysis.head(10).reset_index()
sns.barplot(x='Total_Profit', y='State', data=top_10_states)
plt.title('Top 10 States by Profit')
plt.show()


# In[14]:


# Top 5 Best-Selling Products
top_selling_products = data.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(5)
print("Top 5 Best-Selling Products:")
print(top_selling_products)


# In[15]:


# Top 5 Least-Selling Products
least_selling_products = data.groupby('Sub-Category')['Sales'].sum().sort_values().head(5)
print("\nTop 5 Least-Selling Products:")
print(least_selling_products)


# In[16]:


# Average Sales and Profit per Order
data['Order Profitability'] = data['Profit'] / data['Sales']
avg_sales_per_order = data['Sales'].mean()
avg_profit_per_order = data['Profit'].mean()
avg_order_profitability = data['Order Profitability'].mean()

print(f"Average Sales per Order: {avg_sales_per_order}")
print(f"Average Profit per Order: {avg_profit_per_order}")
print(f"Average Order Profitability: {avg_order_profitability}")


# In[17]:


# Analysis by Ship Mode
ship_mode_analysis = data.groupby('Ship Mode').agg(Total_Sales=('Sales', 'sum'),
                                                    Total_Profit=('Profit', 'sum'),
                                                    Avg_Delivery_Profitability=('Order Profitability', 'mean'))
print(ship_mode_analysis)


# In[18]:


# Pivot table for Sales by State and Category
pivot_sales_state_category = pd.pivot_table(data, values='Sales', index='State', columns='Category', aggfunc='sum')

# Heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(pivot_sales_state_category, cmap="GnBu")
plt.title('Heatmap of Sales by State and Category')
plt.show()


# In[19]:


# Scatter plot of Sales vs. Profit
sns.scatterplot(x='Sales', y='Profit', data=data)
plt.title('Sales vs. Profit')
plt.show()


# In[20]:


# List all numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
print(numeric_columns)


# In[21]:


# Calculate the correlation matrix for numeric columns only
correlation_matrix = data[numeric_columns].corr()

# Print the correlation with 'Profit', sorted
print(correlation_matrix['Profit'].sort_values(ascending=False))


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting features and target variable
X = data[['Sales', 'Quantity', 'Discount']]
y = data['Profit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)


# In[24]:


# Calculate the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# # Refining the Regression Analysis Based on Correlation Findings
# 
# Your correlation findings indicate:<br>
# A moderate positive correlation between Sales and Profit (0.479064).<br>
# A very weak positive correlation between Quantity and Profit (0.066253).<br>
# A weak negative correlation between Discount and Profit (-0.219487).<br>
# The negative R-squared value in the model's performance metrics suggests that the model fits the data worse than a horizontal line (i.e., a simple mean of the observed Profit). This outcome is highly unusual in practice and indicates that the model's assumptions or the selected features may not be appropriate for predicting Profit.

# #### Given the weak correlation between Quantity and Profit, and considering the negative R-squared value, it's worth evaluating if excluding Quantity and focusing on Sales and Discount improves the model. Additionally, exploring non-linear relationships or interactions between features might be beneficial.
# 
# Next step involves developing a More Complex Model<br>
# Given the potential non-linear relationship, let's explore a more complex model. Polynomial features for Sales and Discount could capture the non-linear effects on Profit.

# In[25]:


from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(data[['Sales', 'Discount']])

# Split the data
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = model_poly.predict(X_test_poly)


# In[26]:


print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_poly))
print("R-squared (R2):", r2_score(y_test, y_pred_poly))


# #### Polynomial Regression Model Improvement
# After identifying the linear model's limitations, I applied a polynomial model that significantly improved the predictive accuracy for profit, showing an R-squared value of 0.716, suggesting that incorporating non-linear features could capture the complex dynamics affecting profitability more effectively.

# In[27]:


# Sales vs. Profit
sns.scatterplot(x='Sales', y='Profit', data=data)
plt.title('Sales vs. Profit')
plt.show()



# In[28]:


# Discount vs. Profit
sns.scatterplot(x='Discount', y='Profit', data=data)
plt.title('Discount vs. Profit')
plt.show()


# In[29]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Profits')
plt.ylabel('Predicted Profits')
plt.title('Actual vs. Predicted Profits')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()


# In[30]:


sns.histplot((y_test - y_pred), bins=50, kde=True)
plt.title('Error Distribution')
plt.show()


# In[31]:


# Sales vs. Profit
sns.scatterplot(data=data, x='Sales', y='Profit', hue='Discount', palette="coolwarm")
plt.title('Sales vs. Profit by Discount Levels')
plt.show()


# In[32]:


plt.scatter(y_test, y_pred_poly)
plt.xlabel('Actual Profits')
plt.ylabel('Predicted Profits')
plt.title('Actual vs. Predicted Profits (Polynomial Model)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()


# In[33]:


sns.histplot(y_test - y_pred_poly, bins=50, kde=True)
plt.title('Error Distribution (Polynomial Model)')
plt.xlabel('Prediction Error')
plt.show()


# Based on the comprehensive analysis we conducted, including correlation analysis, linear regression, polynomial feature enhancement, and various data visualizations, let's synthesize the insights gained and formulate recommendations to increase profit for the Sample Superstore.
# 
# ## Insights
# 
# 1. Sales and Profit Correlation: There's a moderate positive correlation between Sales and Profit, indicating that higher sales generally lead to higher profits. However, this relationship isn't linear, as shown by the improved performance of the polynomial model.
# 
# 2. Impact of Discounts: The negative correlation between Discount and Profit suggests that higher discounts may reduce profitability. The visualization of Sales vs. Profit by Discount levels further supports this, showing that higher discounts don't always correspond to increased profitability.
# 
# 3. Sub-Category Performance: Certain sub-categories, like Tables and Bookcases, show negative profitability, which drags down the overall profit in the Furniture category. Conversely, sub-categories like Copiers in the Technology category are highly profitable.
# 
# 4. Geographical Variations: The analysis by state revealed that some states, such as Texas and Ohio, are significantly less profitable than others. This suggests geographical market performance varies widely, possibly due to different competitive landscapes, market saturation, or operational inefficiencies.
# 
# 5. Segment Analysis: Different customer segments exhibit different levels of profitability, with the Consumer segment being the most profitable, followed by Corporate and Home Office. This indicates varying levels of effectiveness in the company's engagement strategies across segments.
# 
# ## Recommendations
# 
# 1. Review Discount Strategies: Given the negative impact of discounts on profit, it's crucial to review current discount strategies. Consider limiting discounts on low-margin products or in regions where discounts do not lead to volume sales that compensate for the lower margin.
# 
# 2. Focus on High-Profit Categories and Sub-Categories: Increase focus on Technology products, especially high-margin items like Copiers. Consider reallocating marketing budgets to promote these higher-margin products more aggressively.
# 
# 3. Optimize Product Mix in Underperforming States: For states like Texas and Ohio, where profitability is low, analyze the product mix and customer buying behavior. Tailoring the product offering to match local demand and optimizing pricing strategies could improve profitability.
# 
# 4. Improve Operational Efficiency: Investigate the supply chain and operational processes in underperforming geographical areas. Reducing operational costs through improved logistics, better supplier negotiations, and inventory management can help increase profit margins.
# 
# 5. Segment-Specific Marketing Strategies: Develop targeted marketing strategies for different customer segments to increase engagement and conversion. For segments like Home Office, which has the lowest profitability, tailored promotions, and product recommendations based on segment-specific needs and buying patterns could enhance profitability.
# 
# 6. Data-Driven Pricing Strategy: Employ a more dynamic pricing strategy that considers product demand, competition, and inventory levels to optimize profitability. Leveraging data analytics to adjust prices in real-time could help maximize margins, especially for high-demand or exclusive products.
# 
# 7. Continuous Monitoring and Analysis: Regularly monitor the market, customer behavior, and competitive landscape to quickly adapt strategies. Use data analytics to identify trends, opportunities for growth, and areas of concern to make informed decisions.
# 
# ### By implementing these recommendations, the owner of store can aim to increase its profitability through strategic discounts, optimized product offerings, targeted marketing, operational efficiencies, and dynamic pricing strategies. Continuous analysis and adaptation will be key to navigating market changes and achieving sustained profitability.

# # THANK YOU
