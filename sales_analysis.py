

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# creating dataset
np.random.seed(42)
n = 10000

categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books']
products = {
    'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch'],
    'Clothing':    ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
    'Home & Kitchen': ['Blender', 'Cookware Set', 'Bedsheet', 'Air Purifier', 'Coffee Maker'],
    'Sports':      ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Cycle', 'Protein Powder'],
    'Books':       ['Fiction Novel', 'Self-Help', 'Science', 'Biography', 'Children']
}
price_range = {
    'Electronics': (5000, 80000),
    'Clothing':    (300, 5000),
    'Home & Kitchen': (500, 15000),
    'Sports':      (200, 20000),
    'Books':       (100, 800)
}
regions = ['North', 'South', 'East', 'West', 'Central']
region_weights = [0.25, 0.20, 0.20, 0.22, 0.13]

start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n)]
cat_choices = np.random.choice(categories, n, p=[0.30, 0.25, 0.20, 0.15, 0.10])

rows = []
for i in range(n):
    cat = cat_choices[i]
    product = np.random.choice(products[cat])
    lo, hi = price_range[cat]
    price = round(np.random.uniform(lo, hi), 2)
    qty = np.random.randint(1, 6)
    discount = np.random.choice([0, 5, 10, 15, 20], p=[0.40, 0.25, 0.20, 0.10, 0.05])
    revenue = round(price * qty * (1 - discount / 100), 2)
    region = np.random.choice(regions, p=region_weights)
    rows.append({
        'order_id':  f'ORD{100000+i}',
        'date':      dates[i],
        'category':  cat,
        'product':   product,
        'price':     price,
        'quantity':  qty,
        'discount':  discount,
        'revenue':   revenue,
        'region':    region
    })

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['quarter'] = df['date'].dt.quarter


df.to_csv('sales_data.csv', index=False)
print(" Dataset created: sales_data.csv")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}\n")

# analysis
print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)
print(f"\nTotal Orders    : {len(df):,}")
print(f"Total Revenue   : ₹{df['revenue'].sum():,.2f}")
print(f"Avg Order Value : ₹{df['revenue'].mean():,.2f}")
print(f"Date Range      : {df['date'].min().date()} → {df['date'].max().date()}")

print("\n Revenue by Category:")
cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
for cat, rev in cat_rev.items():
    print(f"   {cat:<20} ₹{rev:>15,.2f}")

print("\n  Revenue by Region:")
reg_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
for reg, rev in reg_rev.items():
    print(f"   {reg:<10} ₹{rev:>15,.2f}")

print("\n Revenue by Quarter:")
q_rev = df.groupby('quarter')['revenue'].sum()
for q, rev in q_rev.items():
    print(f"   Q{q}  ₹{rev:>15,.2f}")

#plotting
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

fig = plt.figure(figsize=(18, 14))
fig.suptitle('E-Commerce Sales Analysis Dashboard — 2024', fontsize=20, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1: Monthly Revenue Trend
ax1 = fig.add_subplot(gs[0, :])
monthly = df.groupby('month')['revenue'].sum().reset_index()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly['month_name'] = [month_labels[m-1] for m in monthly['month']]
ax1.plot(monthly['month_name'], monthly['revenue']/1e6, marker='o', color='#2196F3',
         linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
ax1.fill_between(range(len(monthly)), monthly['revenue']/1e6, alpha=0.15, color="#74BBF6")
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['month_name'])
ax1.set_title('Monthly Revenue Trend (₹ Million)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Revenue (₹M)')

# Chart 2: Revenue by Category (Bar)
ax2 = fig.add_subplot(gs[1, 0])
cat_rev_sorted = df.groupby('category')['revenue'].sum().sort_values()
bars = ax2.barh(cat_rev_sorted.index, cat_rev_sorted.values/1e6, color=colors)
ax2.set_title('Revenue by Category (₹M)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Revenue (₹M)')

# Chart 3: Revenue by Region (Pie)
ax3 = fig.add_subplot(gs[1, 1])
reg_data = df.groupby('region')['revenue'].sum()
ax3.pie(reg_data.values, labels=reg_data.index, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.82)
ax3.set_title('Revenue Share by Region', fontsize=11, fontweight='bold')

# Chart 4: Orders by Quarter
ax4 = fig.add_subplot(gs[1, 2])
q_orders = df.groupby('quarter').size()
ax4.bar([f'Q{q}' for q in q_orders.index], q_orders.values, color=colors[:4], edgecolor='white', linewidth=1.5)
ax4.set_title('Orders by Quarter', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Orders')

# Chart 5: Discount vs Revenue (Scatter)
ax5 = fig.add_subplot(gs[2, 0])
sample = df.sample(500, random_state=42)
scatter_colors = [colors[categories.index(c)] for c in sample['category']]
ax5.scatter(sample['discount'], sample['revenue'], c=scatter_colors, alpha=0.5, s=30)
ax5.set_title('Discount vs Revenue', fontsize=11, fontweight='bold')
ax5.set_xlabel('Discount (%)')
ax5.set_ylabel('Revenue (₹)')

# Chart 6: Top 10 Products
ax6 = fig.add_subplot(gs[2, 1:])
top_products = df.groupby('product')['revenue'].sum().nlargest(10).sort_values()
ax6.barh(top_products.index, top_products.values/1e6, color='#2196F3', alpha=0.8)
ax6.set_title('Top 10 Products by Revenue (₹M)', fontsize=11, fontweight='bold')
ax6.set_xlabel('Revenue (₹M)')

plt.savefig('sales_dashboard.png', dpi=150, bbox_inches='tight')
print("\nDashboard saved: sales_dashboard.png")

#insights
print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
top_cat = cat_rev.idxmax()
top_region = reg_rev.idxmax()
q4_share = (df[df['quarter']==4]['revenue'].sum() / df['revenue'].sum()) * 100
top_product = df.groupby('product')['revenue'].sum().idxmax()
print(f"  1. '{top_cat}' is the highest-grossing category.")
print(f"  2. '{top_region}' region leads in revenue with ₹{reg_rev.max():,.0f}.")
print(f"  3. Q4 accounts for {q4_share:.1f}% of annual revenue — holiday effect.")
print(f"  4. '{top_product}' is the best-selling product overall.")
avg_discount_rev = df.groupby('discount')['revenue'].mean()
print(f"  5. Orders with 0% discount have highest avg revenue: ₹{avg_discount_rev[0]:,.0f}")

print("\n Project 1 Complete!")
