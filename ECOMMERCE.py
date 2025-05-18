#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector


db = mysql.connector.connect(host = "localhost",
                            username = "root",
                            password = "Maha@143",
                            database = "ecommerce")

cur = db.cursor()


# In[4]:


import pandas as pd
import mysql.connector
import os

# List of CSV files and their corresponding table names
csv_files = [
    ('customers.csv', 'customers'),
    ('orders.csv', 'orders'),
    ('geolocation.csv', 'geolocation'),
    ('products.csv', 'products'),
    ('sellers.csv', 'sellers'),
    ('payments.csv', 'payments'),
    ('order_items.csv', 'order_items')# Added payments.csv for specific handling
]

# Connect to the MySQL database
conn = mysql.connector.connect(
    host = "localhost",
                            username = "root",
                            password = "Maha@143",
                            database = "ecommerce"
)
cursor = conn.cursor()

# Folder containing the CSV files
folder_path = 'C:/Users/91704/Desktop/DATA PROJECTS/E-COMMERCE'

def get_sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'DATETIME'
    else:
        return 'TEXT'

for csv_file, table_name in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Replace NaN with None to handle SQL NULL
    df = df.where(pd.notnull(df), None)
    
    # Debugging: Check for NaN values
    print(f"Processing {csv_file}")
    print(f"NaN values before replacement:\n{df.isnull().sum()}\n")

    # Clean column names
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]

    # Generate the CREATE TABLE statement with appropriate data types
    columns = ', '.join([f'`{col}` {get_sql_type(df[col].dtype)}' for col in df.columns])
    create_table_query = f'CREATE TABLE IF NOT EXISTS `{table_name}` ({columns})'
    cursor.execute(create_table_query)

    # Insert DataFrame data into the MySQL table
    for _, row in df.iterrows():
        # Convert row to tuple and handle NaN/None explicitly
        values = tuple(None if pd.isna(x) else x for x in row)
        sql = f"INSERT INTO `{table_name}` ({', '.join(['`' + col + '`' for col in df.columns])}) VALUES ({', '.join(['%s'] * len(row))})"
        cursor.execute(sql, values)

    # Commit the transaction for the current CSV file
    conn.commit()

# Close the connection
conn.close()


# # 1. List all unique cities where customers are located.

# In[33]:


query = """ select distinct customer_city from customers """

cur.execute(query)

data = cur.fetchall()

data
 


# # 2. Count the number of orders placed in 2017.

# In[7]:


query = """ select count(order_id) from orders where year(order_purchase_timestamp) = 2017 """

cur.execute(query)

data = cur.fetchall()

"total orders placed in 2017 are", data[0][0]


# # 3. Find the total sales per category.

# In[8]:


query = """ select upper(products.product_category) category, 
round(sum(payments.payment_value),2) sales
from products join order_items 
on products.product_id = order_items.product_id
join payments 
on payments.order_id = order_items.order_id
group by category
"""

cur.execute(query)

data = cur.fetchall()

df = pd.DataFrame(data, columns = ["Category", "Sales"])
df


# # 4. Calculate the percentage of orders that were paid in instalments.

# In[9]:


query = """ select ((sum(case when payment_installments >= 1 then 1
else 0 end))/count(*))*100 from payments
"""

cur.execute(query)

data = cur.fetchall()

"the percentage of orders that were paid in installments is", data[0][0]


# # 5. Count the number of customers from each state. 

# In[10]:


query = """ select customer_state ,count(customer_id)
from customers group by customer_state
"""

cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns = ["state", "customer_count" ])
df = df.sort_values(by = "customer_count", ascending= False)

plt.figure(figsize = (8,3))
plt.bar(df["state"], df["customer_count"])
plt.xticks(rotation = 90)
plt.xlabel("states")
plt.ylabel("customer_count")
plt.title("Count of Customers by States")
plt.show()


# # 1. Calculate the number of orders per month in 2018.

# In[30]:


query = """ select monthname(order_purchase_timestamp) months, count(order_id) order_count
from orders where year(order_purchase_timestamp) = 2018
group by months
"""

cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns = ["months", "order_count"])
o = ["January", "February","March","April","May","June","July","August","September","October"]

import seaborn as sns
import pandas as pd

ax = sns.barplot(x = df["months"],y =  df["order_count"], data = df, order = o, color = "red")
plt.xticks(rotation = 45)
ax.bar_label(ax.containers[0])
plt.title("Count of Orders by Months is 2018")

plt.show()


# # 2. Find the average number of products per order, grouped by customer city.

# In[14]:


query = """with count_per_order as 
(select orders.order_id, orders.customer_id, count(order_items.order_id) as oc
from orders join order_items
on orders.order_id = order_items.order_id
group by orders.order_id, orders.customer_id)

select customers.customer_city, round(avg(count_per_order.oc),2) average_orders
from customers join count_per_order
on customers.customer_id = count_per_order.customer_id
group by customers.customer_city order by average_orders desc
"""

cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data,columns = ["customer city", "average products/order"])
df.head(10)


# # 3. Calculate the percentage of total revenue contributed by each product category.

# In[15]:


query = """select upper(products.product_category) category, 
round((sum(payments.payment_value)/(select sum(payment_value) from payments))*100,2) sales_percentage
from products join order_items 
on products.product_id = order_items.product_id
join payments 
on payments.order_id = order_items.order_id
group by category order by sales_percentage desc"""


cur.execute(query)
data = cur.fetchall()
df = pd.DataFrame(data,columns = ["Category", "percentage distribution"])
df.head()


# # 4. Identify the correlation between product price and the number of times a product has been purchased.

# In[16]:


cur = db.cursor()
query = """select products.product_category, 
count(order_items.product_id),
round(avg(order_items.price),2)
from products join order_items
on products.product_id = order_items.product_id
group by products.product_category"""

cur.execute(query)
data = cur.fetchall()
df = pd.DataFrame(data,columns = ["Category", "order_count","price"])

arr1 = df["order_count"]
arr2 = df["price"]

a = np.corrcoef([arr1,arr2])
print("the correlation is", a[0][-1])


# # 5. Calculate the total revenue generated by each seller, and rank them by revenue.

# In[29]:


query = """ select *, dense_rank() over(order by revenue desc) as rn from
(select order_items.seller_id, sum(payments.payment_value)
revenue from order_items join payments
on order_items.order_id = payments.order_id
group by order_items.seller_id) as a """

cur.execute(query)
data = cur.fetchall()

# Create DataFrame with proper column names
df = pd.DataFrame(data, columns=["seller_id", "revenue", "rank"])

# Sort by revenue in descending order and take top results
df = df.sort_values("revenue", ascending=False).head()

# Create the plot
plt.figure(figsize=(10, 6))  # Set a specific figure size
sns.barplot(x="seller_id", y="revenue", data=df)


import seaborn as sns
import pandas as pd

# Customize plot
plt.title("Top Sellers by Revenue")
plt.xlabel("Seller ID")
plt.ylabel("Revenue")
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust layout to prevent cutting off labels

# Show the plot
plt.show()


# # 1. Calculate the moving average of order values for each customer over their order history.

# In[36]:


query = """select customer_id, order_purchase_timestamp, payment,
avg(payment) over(partition by customer_id order by order_purchase_timestamp
rows between 2 preceding and current row) as mov_avg
from
(select orders.customer_id, orders.order_purchase_timestamp, 
payments.payment_value as payment
from payments join orders
on payments.order_id = orders.order_id) as a"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Execute the query
cur.execute(query)
data = cur.fetchall()

# Create DataFrame with column names
df = pd.DataFrame(data, columns=['customer_id', 'order_purchase_timestamp', 'payment', 'moving_average'])

# Convert timestamp to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Sort the dataframe by customer_id and timestamp
df = df.sort_values(['customer_id', 'order_purchase_timestamp'])

# Take the first 10 unique customers
top_customers = df['customer_id'].unique()[:10]

# Create a figure with subplots for each customer
plt.figure(figsize=(20, 15))

for i, customer in enumerate(top_customers, 1):
    # Filter data for the specific customer
    customer_data = df[df['customer_id'] == customer]
    
    # Create subplot
    plt.subplot(5, 2, i)
    
    # Plot actual payments
    plt.plot(customer_data['order_purchase_timestamp'], 
             customer_data['payment'], 
             marker='o', 
             label='Payment', 
             color='blue')
    
    # Plot moving average
    plt.plot(customer_data['order_purchase_timestamp'], 
             customer_data['moving_average'], 
             marker='x', 
             label='3-Order Moving Average', 
             color='red')
    
    plt.title(f'Customer {customer} - Payments and Moving Average')
    plt.xlabel('Order Timestamp')
    plt.ylabel('Payment Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Optional: Print the first 10 rows of the dataframe for reference
print(df.head(10))


# # 2. Calculate the cumulative sales per month for each year.

# In[39]:


query = """select years, months , payment, sum(payment)
over(order by years, months) cumulative_sales from 
(select year(orders.order_purchase_timestamp) as years,
month(orders.order_purchase_timestamp) as months,
round(sum(payments.payment_value),2) as payment from orders join payments
on orders.order_id = payments.order_id
group by years, months order by years, months) as a

"""
cur.execute(query)
data = cur.fetchall()
df = pd.DataFrame(data, columns=["year", "month", "monthly_payment", "cumulative_sales"])

# Show first 10 rows
print(df.head(5))

# Create a new 'date' column for proper x-axis labeling
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

# Plotting monthly payment and cumulative sales
plt.figure(figsize=(14, 6))

# Barplot for monthly payments
sns.barplot(x="date", y="monthly_payment", data=df, color="skyblue", label="Monthly Payment")

# Lineplot for cumulative sales
sns.lineplot(x="date", y="cumulative_sales", data=df, color="red", label="Cumulative Sales", linewidth=2)

plt.title("Monthly Payments and Cumulative Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Payment Value")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# # 3. Calculate the year-over-year growth rate of total sales.

# In[40]:


query = """with a as(select year(orders.order_purchase_timestamp) as years,
round(sum(payments.payment_value),2) as payment from orders join payments
on orders.order_id = payments.order_id
group by years order by years)

select years, ((payment - lag(payment, 1) over(order by years))/
lag(payment, 1) over(order by years)) * 100 from a"""

# Execute the query
cur.execute(query)
data = cur.fetchall()

# Create DataFrame with column names
df = pd.DataFrame(data, columns=['Year', 'YoY_Growth_Percentage'])

# Remove any potential infinity or NaN values
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Create the visualization
plt.figure(figsize=(12, 6))

# Bar plot for Year-over-Year Growth
plt.subplot(1, 2, 1)
sns.barplot(x='Year', y='YoY_Growth_Percentage', data=df, palette='coolwarm')
plt.title('Year-over-Year Payment Growth')
plt.xlabel('Year')
plt.ylabel('Growth Percentage')
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at 0%
plt.xticks(rotation=45)

# Line plot for Year-over-Year Growth
plt.subplot(1, 2, 2)
plt.plot(df['Year'], df['YoY_Growth_Percentage'], marker='o', linestyle='-', color='green')
plt.title('Year-over-Year Payment Growth (Line)')
plt.xlabel('Year')
plt.ylabel('Growth Percentage')
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at 0%
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print the dataframe for reference
print("Yearly Payment Growth Data:")
print(df)

# Calculate some additional insights
print("\nAdditional Insights:")
print(f"Average Year-over-Year Growth: {df['YoY_Growth_Percentage'].mean():.2f}%")
print(f"Maximum Year-over-Year Growth: {df['YoY_Growth_Percentage'].max():.2f}%")
print(f"Minimum Year-over-Year Growth: {df['YoY_Growth_Percentage'].min():.2f}%")


# # 4. Calculate the retention rate of customers, defined as the percentage of customers who make another purchase within 6 months of their first purchase.

# In[42]:


query = """with a as (select customers.customer_id,
min(orders.order_purchase_timestamp) first_order
from customers join orders
on customers.customer_id = orders.customer_id
group by customers.customer_id),

b as (select a.customer_id, count(distinct orders.order_purchase_timestamp) next_order
from a join orders
on orders.customer_id = a.customer_id
and orders.order_purchase_timestamp > first_order
and orders.order_purchase_timestamp < 
date_add(first_order, interval 6 month)
group by a.customer_id) 

select 100 * (count( distinct a.customer_id)/ count(distinct b.customer_id)) 
from a left join b 
on a.customer_id = b.customer_id ;"""

cur.execute(query)
data = cur.fetchall()

data


# # 5. Identify the top 3 customers who spent the most money in each year.

# In[34]:


query = """select years, customer_id, payment, d_rank
from
(select year(orders.order_purchase_timestamp) years,
orders.customer_id,
sum(payments.payment_value) payment,
dense_rank() over(partition by year(orders.order_purchase_timestamp)
order by sum(payments.payment_value) desc) d_rank
from orders join payments 
on payments.order_id = orders.order_id
group by year(orders.order_purchase_timestamp),
orders.customer_id) as a
where d_rank <= 3 ;"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data, columns=["years", "customer_id", "payment", "rank"])

import seaborn as sns
import pandas as pd

# Fix: convert 'years' to string for hue compatibility
df["years"] = df["years"].astype(str)

sns.barplot(x="customer_id", y="payment", data=df, hue="years")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



# In[ ]:




