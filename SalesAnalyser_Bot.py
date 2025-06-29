import streamlit as st
import os
import mysql.connector as sql
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from datetime import datetime
from langchain_ollama import OllamaLLM, OllamaEmbeddings

st.set_page_config(
        page_title="SalesAnalyser - Financial Analysis",
        page_icon="📈",
        layout="wide"
)

#=========================================================
#STEP 1: INSTANTIATION
#=========================================================

try:
    model = OllamaLLM(model="granite3.3:8b", temperature=0.1)
    embeddings_model = OllamaEmbeddings(model="granite-embedding:30m")
    #st.success("Models loaded successfully!")

except Exception as e:
    st.error(f"Failed to connect to Ollama or load models: {e}")
    st.stop()

#===============================================================================
#STEP 2: RAG IMPLEMENTATION FOR QUESTION-SQL PAIRS
#=============================================================================

class SQLExampleRetriever:
    '''
    1. Load Question-SQL pairs from the CSV file
    2. Create new embeddings for the questions if they do not exist, else manage
    cached ones
    3. Use cosine similarity to find the most relevant examples for a given query
    4. Format the examples for prompt template
    '''
    """Implement RAG to store and retrieve similar question-SQL pairs"""
    def __init__(self, csv_file_path, embeddings_model, cache_file="sql_embeddings_cache.pkl"):
        self.csv_file_path = csv_file_path
        self.embeddings_model = embeddings_model
        self.cache_file = cache_file
        self.examples_df = None
        self.question_embeddings = None
        self.load_examples()
        
    def load_examples(self):
        """Load the CSV file containing question-SQL pairs"""
        try:
            self.examples_df = pd.read_csv(self.csv_file_path)
            # Ensure required columns exist
            required_columns = ['Question', 'SQL'] 
            if not all(col in self.examples_df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {required_columns}")
                st.error(f"Found columns: {list(self.examples_df.columns)}")
                return
            
            self.load_or_create_embeddings()
            
        except FileNotFoundError:
            st.error("Please ensure your question-SQL pairs CSV file is in the same directory")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    
    def load_or_create_embeddings(self):
        """Load cached embeddings or create new ones"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if len(cache_data['embeddings']) == len(self.examples_df):
                        self.question_embeddings = cache_data['embeddings']
                        #st.info("Loaded cached embeddings")
                        return
            except Exception as e:
                st.warning(f"Could not load cached embeddings: {e}")
        
        # Create new embeddings
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all questions in the dataset"""
        #st.info("Creating embeddings for question-SQL pairs... This may take a few minutes.")
        
        questions = self.examples_df['Question'].tolist()
        embeddings = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        for i, question in enumerate(questions):
            try:
                embedding = self.embeddings_model.embed_query(question)
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / len(questions))
            except Exception as e:
                st.error(f"Error creating embedding for question {i}: {e}")
                embeddings.append([0] * 384)  # Fallback embedding
        
        self.question_embeddings = np.array(embeddings)
        
        # Cache the embeddings
        try:
            cache_data = {
                'embeddings': self.question_embeddings,
                'csv_hash': hash(str(self.examples_df.values.tobytes()))
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            #st.success("Embeddings created and cached successfully!")
        except Exception as e:
            st.warning(f"Could not cache embeddings: {e}")
    
    def retrieve_similar_examples(self, query, top_k=5):
        """Retrieve the most similar examples for a given query"""
        if self.examples_df is None or self.question_embeddings is None:
            return []
        
        try:
            # Create embedding for the query
            query_embedding = self.embeddings_model.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
            
            # Get top-k most similar examples
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_examples = []
            for idx in top_indices:
                example = {
                    'question': self.examples_df.iloc[idx]['Question'],
                    'sql': self.examples_df.iloc[idx]['SQL'],
                    'similarity': similarities[idx]
                }
                similar_examples.append(example)
            
            return similar_examples
            
        except Exception as e:
            st.error(f"Error retrieving similar examples: {e}")
            return []
    
    def format_examples_for_prompt(self, examples):
        """Format retrieved examples for inclusion in the prompt"""
        if not examples:
            return ""
        
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_examples.append(f"{i}. Question: {example['question']}\n   SQL: {example['sql']}")
        
        return "\n\n".join(formatted_examples)

# Initialize the RAG retriever
@st.cache_resource
def initialize_rag_retriever():
    csv_file_path = "English_SQL.csv" 
    return SQLExampleRetriever(csv_file_path, embeddings_model)

# Only initialize if CSV file exists
if os.path.exists("English_SQL.csv"):
    rag_retriever = initialize_rag_retriever()
else:
    st.warning("CSV file 'English_SQL.csv' not found. Using fallback examples.")
    rag_retriever = None

#==================================================
#STEP 3: DATABASE CONFIGURATION
#==================================================

mysql_config = {
    'host': 'localhost',
    'user': 'root',  
    'password': 'themortalinstruments',
    'database': 'sales'
}

#============================================
#STEP 4: DATABASE INITIALIZATION
#===========================================

def initialize_database():
    mycon = sql.connect( host=mysql_config['host'], user=mysql_config['user'],
                         password=mysql_config['password'] )
    cursor = mycon.cursor()
    
    # Step 1: Create database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_config['database']}")
    cursor.execute(f"USE {mysql_config['database']}")
    
    # Step 2: Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ProductSales (
        OrderID VARCHAR(255) PRIMARY KEY,
        Date DATE,
        Product VARCHAR(255),
        Category VARCHAR(255),
        BaseCost FLOAT,
        SellingPrice FLOAT,
        Quantity FLOAT,
        TotalSales FLOAT,
        CustomerName VARCHAR(255),
        CustomerLocation VARCHAR(50),
        PaymentMethod VARCHAR(50),
        Status VARCHAR(255)
    )
    """)
    
    # Step 3: Check if data needs to be imported by checking count of no.of rows
    cursor.execute("SELECT COUNT(*) FROM ProductSales")
    count = cursor.fetchone()[0]
    
    # if no.of rows==0 (ie table empty in mysql) and file exists
    if count == 0 and os.path.exists("amazon_sales_data_kaggle4.csv"):
        df = pd.read_csv("amazon_sales_data_kaggle4.csv")
        
        # Convert DataFrame to list of tuples for bulk insert
        records = []
        for _, row in df.iterrows():
            date_parts = row['Date'].split('-')
            mysql_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            
            records.append(( row['OrderID'], mysql_date, row['Product'], row['Category'], float(row['BaseCost']), 
                            float(row['SellingPrice']), float(row['Quantity']), float(row['TotalSales']), 
                            row['CustomerName'], row['CustomerLocation'], row['PaymentMethod'], row['Status'] ))
        
        # Bulk insert data
        sql_query = """INSERT INTO ProductSales 
                 (OrderID, Date, Product, Category, BaseCost, SellingPrice, Quantity, TotalSales, CustomerName, CustomerLocation, PaymentMethod, Status) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.executemany(sql_query, records)
        mycon.commit()
        print(f"Imported {len(records)} records from amazon_sales_data_kaggle4.csv")
    
    mycon.close()

#=========================================================
#STEP 5: ENHANCED QUESTION ANALYZER AND VISUALIZATION LOGIC
#=========================================================

class QuestionAnalyzer:
    '''
    1. Identifies what the question is asking about
    2. Determines what to measure (whether its sales, product, quantity etc)
    3. Extracts time periods, categories, locations from questions
    4. Determines if its a direct lookup (no visualization necessary) or comparison query
    (graph can be made)
    '''
    """Analyzes questions to extract subject, filters, and metrics for dual query generation"""
    
    def __init__(self):
        self.subjects = {
            'WHO': r'\b(?:who|which customer|customer|customers)\b',
            'WHICH_PRODUCT': r'\b(?:which product|product|products|what product)\b',
            'WHICH_CATEGORY': r'\b(?:which category|category|categories)\b',
            'WHICH_LOCATION': r'\b(?:which location|location|region|city)\b',
            'WHICH_PAYMENT': r'\b(?:which payment|payment method|payment)\b'
        }
        
        self.metrics = {
            'sales': r'\b(?:sales|revenue|amount|total sales|sales amount)\b',
            'profit': r'\b(?:profit|profits|profitability)\b',
            'quantity': r'\b(?:quantity|units|items|pieces)\b',
            'orders': r'\b(?:orders|transactions|purchases)\b',
            'price': r'\b(?:price|cost|pricing)\b'
        }
        
        self.time_filters = {
            'year': r'\b(?:in |during |for |year )\d{4}\b',
            'month': r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            'period': r'\b(?:in \d{4}|during \d{4}|last year|this year|current year|previous year)\b'
        }
        
        self.comparison_words = r'\b(?:highest|most|lowest|least|smallest|greatest|maximum|minimum|top|bottom)\b'
        
        # Add patterns for direct questions that don't need visualization
        self.direct_question_patterns = [
            r'\bwhat was\b.*\bwhen\b',  # "what was X when Y"
            r'\bwhat is\b.*\bfor\b.*\border\b',  # "what is X for order Y"
            r'\bshow me\b.*\bfor\b.*\border\b',  # "show me X for order Y"
            r'\bget\b.*\bfor\b.*\border\b',  # "get X for order Y"
            r'\bfind\b.*\bfor\b.*\border\b',  # "find X for order Y"
            r'\bwhere\b.*\b(?:orderid|order id)\b',  # questions with specific order ID
            r'\bcorresponding\b.*\border\b',  # "corresponding to order"
            r'\bwhat.*name.*order\b',  # "what customer name for order"
            r'\bwhat.*location.*order\b',  # "what location for order"
            r'\bwhat.*product.*order\b',  # "what product for order"
        ]
    
    def is_direct_question(self, question):
        """Check if the question is a direct lookup that doesn't need visualization"""
        question_lower = question.lower()
        for pattern in self.direct_question_patterns:
            if re.search(pattern, question_lower):
                return True
        return False
    
    def analyze_question(self, question):
        """
        Parse question to extract subject, filters, and metrics
        """
        question_lower = question.lower()
        
        analysis = {
            'subject': None,
            'filters': {
                'time': None,
                'product': None,
                'category': None,
                'location': None
            },
            'metric': 'sales',  # default
            'is_comparison': bool(re.search(self.comparison_words, question_lower)),
            'is_direct_question': self.is_direct_question(question),
            'original_question': question
        }
        
        # Extract subject
        for subject_type, pattern in self.subjects.items():
            if re.search(pattern, question_lower):
                analysis['subject'] = subject_type
                break
        
        # Extract metric
        for metric_type, pattern in self.metrics.items():
            if re.search(pattern, question_lower):
                analysis['metric'] = metric_type
                break
        
        # Extract time filters
        time_match = re.search(r'\b(?:in |during |for |year )(\d{4})\b', question_lower)
        if time_match:
            analysis['filters']['time'] = time_match.group(1)
        
        # Extract month filters
        month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', question_lower)
        if month_match:
            analysis['filters']['month'] = month_match.group(1)
        
        # Extract specific product mentions
        products_mentioned = self._extract_product_mentions(question_lower)
        if products_mentioned:
            analysis['filters']['product'] = products_mentioned[0]  # Take first match
        
        # Extract category mentions
        categories = ['electronics', 'clothing', 'footwear', 'books', 'home appliances']
        for category in categories:
            if category in question_lower:
                analysis['filters']['category'] = category
                break
        
        return analysis
    
    def _extract_product_mentions(self, question_lower):
        """Extract specific product mentions from question"""
        common_products = [
            'running shoes', 'headphones', 'smartwatch', 't-shirt', 'smartphone',
            'book', 'jeans', 'laptop', 'washing machine', 'refrigerator', 'tablet',
            'camera', 'speaker', 'mouse', 'keyboard'
        ]
        
        mentioned_products = []
        for product in common_products:
            if product in question_lower:
                mentioned_products.append(product)
        
        return mentioned_products

class DualQueryGenerator:
    '''
    1. Create two queries - to answer exact question and for creating charts
    2. builds aggregated queries suitable for charts
    '''
    """Generates both answer query and visualization query based on question analysis"""
    
    def __init__(self):
        self.analyzer = QuestionAnalyzer()
    
    def generate_queries(self, question, original_sql):
        """Generate both answer query (specific) and visualization query (broad)"""
        analysis = self.analyzer.analyze_question(question)
        answer_query = original_sql # Answer query is the original SQL
        viz_query = self._generate_visualization_query(analysis) # Generate visualization query based on analysis
        
        return {
            'answer_query': answer_query,
            'visualization_query': viz_query,
            'analysis': analysis
        }
    
    def _generate_visualization_query(self, analysis):
        """
        Generate comprehensive visualization query based on analysis
        """
        subject = analysis['subject']
        metric = analysis['metric']
        filters = analysis['filters']
        
        # Base query components
        select_clause = ""
        from_clause = "FROM ProductSales"
        where_clauses = []
        group_by_clause = ""
        order_by_clause = ""
        
        # Determine SELECT and GROUP BY based on subject
        if subject == 'WHO':
            select_clause = "SELECT CustomerName"
            group_by_clause = "GROUP BY CustomerName"
            order_by_col = "CustomerName"
        elif subject == 'WHICH_PRODUCT':
            select_clause = "SELECT Product"
            group_by_clause = "GROUP BY Product"
            order_by_col = "Product"
        elif subject == 'WHICH_CATEGORY':
            select_clause = "SELECT Category"
            group_by_clause = "GROUP BY Category"
            order_by_col = "Category"
        elif subject == 'WHICH_LOCATION':
            select_clause = "SELECT CustomerLocation"
            group_by_clause = "GROUP BY CustomerLocation"
            order_by_col = "CustomerLocation"
        elif subject == 'WHICH_PAYMENT':
            select_clause = "SELECT PaymentMethod"
            group_by_clause = "GROUP BY PaymentMethod"
            order_by_col = "PaymentMethod"
        else:
            # Default fallback
            select_clause = "SELECT Product"
            group_by_clause = "GROUP BY Product"
            order_by_col = "Product"
        
        # Add metric to SELECT clause
        if metric == 'sales':
            select_clause += ", SUM(TotalSales) AS TotalSales"
            order_by_clause = "ORDER BY TotalSales DESC"
        elif metric == 'profit':
            select_clause += ", SUM(TotalSales - (BaseCost * Quantity)) AS TotalProfit"
            order_by_clause = "ORDER BY TotalProfit DESC"
        elif metric == 'quantity':
            select_clause += ", SUM(Quantity) AS TotalQuantity"
            order_by_clause = "ORDER BY TotalQuantity DESC"
        elif metric == 'orders':
            select_clause += ", COUNT(*) AS OrderCount"
            order_by_clause = "ORDER BY OrderCount DESC"
        elif metric == 'price':
            select_clause += ", AVG(SellingPrice) AS AveragePrice"
            order_by_clause = "ORDER BY AveragePrice DESC"
        else:
            select_clause += ", SUM(TotalSales) AS TotalSales"
            order_by_clause = "ORDER BY TotalSales DESC"
        
        # Apply filters from original question
        if filters['time']:
            where_clauses.append(f"YEAR(Date) = {filters['time']}")
        
        if filters.get('month'):
            month_num = self._get_month_number(filters['month'])
            if month_num:
                where_clauses.append(f"MONTH(Date) = {month_num}")
        
        if filters['product']:
            where_clauses.append(f"LOWER(Product) LIKE LOWER('%{filters['product']}%')")
        
        if filters['category']:
            where_clauses.append(f"LOWER(Category) LIKE LOWER('%{filters['category']}%')")
        
        if filters['location']:
            where_clauses.append(f"LOWER(CustomerLocation) LIKE LOWER('%{filters['location']}%')")
        
        # Construct final query
        query_parts = [select_clause, from_clause]
        
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        if group_by_clause:
            query_parts.append(group_by_clause)
        
        if order_by_clause:
            query_parts.append(order_by_clause)
        
        return " ".join(query_parts)
    
    def _get_month_number(self, month_name):
        """Convert month name to number"""
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        return months.get(month_name.lower())

# Initialize dual query generator
dual_query_generator = DualQueryGenerator()

#=======================================
#STEP 6: ENHANCED CHART GENERATION
#=======================================

def determine_chart_data_enhanced(question, df, columns, analysis):
    '''
    1. Figure out which labels to appear on x-axis and y-axis based on analysis
    2. Create an empty chart with no data (incase reply of SQL query = empty set)
    3. Create line chart with insights
    '''
    """Enhanced chart data determination based on question analysis"""
    chart_config = {
        'x_col': None,
        'y_col': None,
        'title': 'Data Visualization',
        'x_label': '',
        'y_label': ''
    }
    
    if len(df) == 0:
        return chart_config
    
    # Identify columns based on analysis
    subject = analysis.get('subject', '')
    metric = analysis.get('metric', 'sales')
    
    # Determine X-axis based on subject
    if subject == 'WHO':
        chart_config['x_col'] = 'CustomerName'
        chart_config['x_label'] = 'Customer'
    elif subject == 'WHICH_PRODUCT':
        chart_config['x_col'] = 'Product'
        chart_config['x_label'] = 'Product'
    elif subject == 'WHICH_CATEGORY':
        chart_config['x_col'] = 'Category'
        chart_config['x_label'] = 'Category'
    elif subject == 'WHICH_LOCATION':
        chart_config['x_col'] = 'CustomerLocation'
        chart_config['x_label'] = 'Location'
    elif subject == 'WHICH_PAYMENT':
        chart_config['x_col'] = 'PaymentMethod'
        chart_config['x_label'] = 'Payment Method'
    else:
        # Find first categorical column
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if categorical_cols:
            chart_config['x_col'] = categorical_cols[0]
            chart_config['x_label'] = categorical_cols[0]
    
    # Determine Y-axis based on metric
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if metric == 'sales' and 'TotalSales' in df.columns:
        chart_config['y_col'] = 'TotalSales'
        chart_config['y_label'] = 'Total Sales (₹)'
    elif metric == 'profit' and 'TotalProfit' in df.columns:
        chart_config['y_col'] = 'TotalProfit'
        chart_config['y_label'] = 'Total Profit (₹)'
    elif metric == 'quantity' and 'TotalQuantity' in df.columns:
        chart_config['y_col'] = 'TotalQuantity'
        chart_config['y_label'] = 'Total Quantity'
    elif metric == 'orders' and 'OrderCount' in df.columns:
        chart_config['y_col'] = 'OrderCount'
        chart_config['y_label'] = 'Number of Orders'
    elif metric == 'price' and 'AveragePrice' in df.columns:
        chart_config['y_col'] = 'AveragePrice'
        chart_config['y_label'] = 'Average Price (₹)'
    elif numeric_cols:
        chart_config['y_col'] = numeric_cols[0]
        chart_config['y_label'] = numeric_cols[0]
    
    # Set title
    if chart_config['x_col'] and chart_config['y_col']:
        chart_config['title'] = f'{chart_config["y_label"]} by {chart_config["x_label"]}'
    return chart_config

def create_empty_chart():
    """Create an empty chart with a message when no data is relevant for visualization"""
    fig = go.Figure()
    
    fig.add_annotation(
        text="No data relevant for query",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=20, color="gray"),
        align="center"
    )
    
    fig.update_layout(
        title="No data relevant for query",
        height=400,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_enhanced_line_chart(df, chart_config, analysis):
    """Create enhanced line charts with better formatting and insights"""
    if df.empty or not chart_config['x_col'] or not chart_config['y_col']:
        return create_empty_chart()
    
    # Ensure the columns exist in the dataframe
    if chart_config['x_col'] not in df.columns or chart_config['y_col'] not in df.columns:
        return create_empty_chart()
    
    # Sort data for better visualization
    df_sorted = df.sort_values(by=chart_config['y_col'], ascending=False)
    
    # Limit to top 15 entries for better readability
    if len(df_sorted) > 15:
        df_sorted = df_sorted.head(15)
        st.info(f"Showing top 15 entries out of {len(df)} total records for better visualization")
    
    # Create line chart
    fig = px.line(
        df_sorted, 
        x=chart_config['x_col'], 
        y=chart_config['y_col'],
        title=chart_config['title'],
        labels={
            chart_config['x_col']: chart_config['x_label'] or chart_config['x_col'],
            chart_config['y_col']: chart_config['y_label'] or chart_config['y_col']
        },
        markers=True,
        line_shape='linear'
    )
    
    # Customize line appearance
    fig.update_traces(
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8, color='#ff7f0e'),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{chart_config["y_label"]}: %{{y:,.0f}}<br>' +
                     '<extra></extra>'
    )
    
    # Update layout for better appearance
    fig.update_layout(
        showlegend=False,
        height=500,
        xaxis_title=chart_config['x_label'] or chart_config['x_col'],
        yaxis_title=chart_config['y_label'] or chart_config['y_col'],
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        title_x=0.5  # Center the title
    )
    
    # Add grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    if len(df_sorted) > 0:
        with st.expander("📊 Chart Insights", expanded=False):
            top_item = df_sorted.iloc[0]
            bottom_item = df_sorted.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Top {chart_config['x_label']}",
                    top_item[chart_config['x_col']],
                    f"₹{top_item[chart_config['y_col']]:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{top_item[chart_config['y_col']]:,.0f}"
                )
            
            with col2:
                if len(df_sorted) > 1:
                    st.metric(
                        f"Bottom {chart_config['x_label']}",
                        bottom_item[chart_config['x_col']],
                        f"₹{bottom_item[chart_config['y_col']]:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{bottom_item[chart_config['y_col']]:,.0f}"
                    )
            
            with col3:
                total_value = df_sorted[chart_config['y_col']].sum()
                st.metric(
                    f"Total {chart_config['y_label']}",
                    f"₹{total_value:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{total_value:,.0f}"
                )
    
    return fig

def should_create_enhanced_chart(df, question, analysis):
    '''
    1. Do not create a chart if its a direct question type, has a single value
       or single row SQL response or if there are no numeric columns
    2. DO CREATE a chart if it answers a pattern-based question or has aggregation
    '''
    """Enhanced logic to determine if a chart should be created based on analysis"""
    # Don't create charts for direct questions
    if analysis.get('is_direct_question', False):
        return False
    # Always create charts for questions with WHO/WHICH patterns
    if analysis.get('subject') in ['WHO', 'WHICH_PRODUCT', 'WHICH_CATEGORY', 'WHICH_LOCATION', 'WHICH_PAYMENT']:
        return True
    # Always create charts for aggregated data (GROUP BY results)
    if len(df) > 1:
        return True
    # Don't create charts for single-row, single-value results
    if len(df) <= 1 and len(df.columns) <= 1:
        return False
    # Don't create charts if there are no numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) == 0:
        return False
    # Create charts for comparison questions
    if analysis.get('is_comparison', False):
        return True
    return True

#===========================
#STEP 7: SQL EXECUTION
#===========================

def execute_sql_query(query):
    try:
        mycon = sql.connect(**mysql_config)
        cursor = mycon.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        mycon.close()
        
        if not result: # Return empty result flag if no records found
            return "EMPTY_RESULT", columns
        return result, columns
    except Exception as e:
        return [f"Error: {str(e)}"], []

#===========================================
#STEP 8: NATURAL LANGUAGE INTERPRETATION 
#===========================================

def natural_language_interpretation(question, sql_query, columns, result):
    """Enhanced interpretation with empty result handling"""
    
    # Check if result is empty
    if result == "EMPTY_RESULT":
        return "📊 **Analysis Result**: There are no transactions/records satisfying the above question's criteria."
    
    # Check for error results
    if isinstance(result, list) and len(result) > 0 and str(result[0]).startswith("Error:"):
        return f"❌ **Error in query execution**: {result[0]}"
    
    # If no results or empty list
    if not result or len(result) == 0:
        return "📊 **Analysis Result**: There are no transactions/records satisfying the above question's criteria."
    
    try:
        df = pd.DataFrame(result, columns=columns)
        
        # Create context for LLM
        context = f"""
        Question: {question}
        SQL Query: {sql_query}
        
        Results ({len(result)} records):
        {df.to_string(index=False, max_rows=10)}
        """
        
        # Create interpretation prompt
        interpretation_prompt = f"""
        Based on the following sales data query and results, provide a clear, concise natural language interpretation:

        {context}

        Instructions:
        1. Directly answer the user's question based on the data
        2. Highlight key insights and numbers
        3. Use Indian Rupee (₹) format for monetary values
        4. Be specific and factual
        5. Keep the response conversational but professional
        6. If there are multiple results, mention the top entries
        7. Format numbers with commas for readability

        Provide your interpretation:
        """
        
        interpretation = model.invoke(interpretation_prompt) # Get interpretation from LLM
        return f"📊 **Analysis Result**: {interpretation}"
        
    except Exception as e:
        return f"📊 **Analysis Result**: Query executed successfully with {len(result) if result != 'EMPTY_RESULT' else 0} records returned."

#=========================
#STEP 9: ENHANCED CHART CREATION
#=========================

def create_empty_chart_with_message(message="No data relevant for query"):
    """Create an empty chart with a custom message"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=20, color="gray"),
        align="center"
    )
    
    fig.update_layout(
        title=message,
        height=400,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig

def create_enhanced_line_chart(df, chart_config, analysis):
    """
    Create enhanced line charts with better formatting and insights
    """
    # Handle empty DataFrame or missing data
    if df is None or df.empty or not chart_config['x_col'] or not chart_config['y_col']:
        fig = create_empty_chart_with_message("No data available for visualization")
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Ensure the columns exist in the dataframe
    if chart_config['x_col'] not in df.columns or chart_config['y_col'] not in df.columns:
        fig = create_empty_chart_with_message("Required columns not found in data")
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Sort data for better visualization
    df_sorted = df.sort_values(by=chart_config['y_col'], ascending=False)
    
    # Limit to top 15 entries for better readability
    if len(df_sorted) > 15:
        df_sorted = df_sorted.head(15)
        st.info(f"Showing top 15 entries out of {len(df)} total records for better visualization")
    
    # Create line chart
    fig = px.line(
        df_sorted, 
        x=chart_config['x_col'], 
        y=chart_config['y_col'],
        title=chart_config['title'],
        labels={
            chart_config['x_col']: chart_config['x_label'] or chart_config['x_col'],
            chart_config['y_col']: chart_config['y_label'] or chart_config['y_col']
        },
        markers=True,
        line_shape='linear'
    )
    
    # Customize line appearance
    fig.update_traces(
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=8, color='#ff7f0e'),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{chart_config["y_label"]}: %{{y:,.0f}}<br>' +
                     '<extra></extra>'
    )
    
    # Update layout for better appearance
    fig.update_layout(
        showlegend=False,
        height=500,
        xaxis_title=chart_config['x_label'] or chart_config['x_col'],
        yaxis_title=chart_config['y_label'] or chart_config['y_col'],
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        title_x=0.5  # Center the title
    )
    
    # Add grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    if len(df_sorted) > 0:
        with st.expander("📊 Chart Insights", expanded=False):
            top_item = df_sorted.iloc[0]
            bottom_item = df_sorted.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Top {chart_config['x_label']}",
                    top_item[chart_config['x_col']],
                    f"₹{top_item[chart_config['y_col']]:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{top_item[chart_config['y_col']]:,.0f}"
                )
            
            with col2:
                if len(df_sorted) > 1:
                    st.metric(
                        f"Bottom {chart_config['x_label']}",
                        bottom_item[chart_config['x_col']],
                        f"₹{bottom_item[chart_config['y_col']]:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{bottom_item[chart_config['y_col']]:,.0f}"
                    )
            
            with col3:
                total_value = df_sorted[chart_config['y_col']].sum()
                st.metric(
                    f"Total {chart_config['y_label']}",
                    f"₹{total_value:,.0f}" if 'Sales' in chart_config['y_col'] or 'Profit' in chart_config['y_col'] else f"{total_value:,.0f}"
                )
    
    return fig

#=========================
#STEP 10: GET SQL QUERY FUNCTION (PLACEHOLDER - REPLACE WITH YOUR EXISTING FUNCTION)
#=========================

def get_sql_query(question):
    """
    Placeholder for your existing get_sql_query function.
    Replace this with your actual implementation that uses RAG retriever and LLM.
    """
    # This is a placeholder - use your existing get_sql_query implementation
    try:
        # Get similar examples from RAG if available
        if rag_retriever:
            similar_examples = rag_retriever.retrieve_similar_examples(question, top_k=3)
            examples_text = rag_retriever.format_examples_for_prompt(similar_examples)
        else:
            examples_text = ""  # Fallback examples
        
        # Create prompt template 
        prompt_template = f"""
            You are an expert SQL query generator for a MySQL database containing sales data. 

            DATABASE SCHEMA:
            Table: ProductSales
            Columns:
            - OrderID (VARCHAR): Unique order identifier
            - Date (DATE): Order date in YYYY-MM-DD format
            - Product (VARCHAR): Product name
            - Category (VARCHAR): Product category
            - BaseCost (FLOAT): Cost price of the product
            - SellingPrice (FLOAT): Selling price of the product
            - Quantity (FLOAT): Quantity sold
            - TotalSales (FLOAT): Total sales amount (SellingPrice * Quantity)
            - CustomerName (VARCHAR): Customer name
            - CustomerLocation (VARCHAR): Customer location/city
            - PaymentMethod (VARCHAR): Payment method used
            - Status (VARCHAR): Order status

            SIMILAR QUERY EXAMPLES:
            {similar_examples}

            QUERY GENERATION RULES:
            1. Generate ONLY the SQL query without any explanation
            2. Use proper MySQL syntax
            3. For date filtering, use YEAR(), MONTH(), DAY() functions
            4. For text searches, use LIKE with % wildcards and LOWER() for case-insensitive matching
            5. For profit calculations: (TotalSales - (BaseCost * Quantity))
            6. Always use appropriate aggregate functions (SUM, COUNT, AVG, MAX, MIN)
            7. Use ORDER BY for ranking questions (highest, lowest, top, etc.)
            8. Use LIMIT for "top N" questions
            9. Use GROUP BY when aggregating data by categories

            EXAMPLES OF QUERY PATTERNS:
            - For "which customer": SELECT CustomerName, ... GROUP BY CustomerName
            - For "which product": SELECT Product, ... GROUP BY Product  
            - For "which category": SELECT Category, ... GROUP BY Category
            - For "highest/most": ORDER BY ... DESC LIMIT 1
            - For "lowest/least": ORDER BY ... ASC LIMIT 1
            - For "top 5": ORDER BY ... DESC LIMIT 5
            - For year filtering: WHERE YEAR(Date) = 2023
            - For month filtering: WHERE MONTH(Date) = 6

            Question: {question}

            SQL Query:"""
        
        # Get SQL from LLM
        sql_query = model.invoke(prompt_template)
        
        # Clean the SQL query
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
        
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

def main():
    # Initialize database
    try:
        initialize_database()
        #st.success("✅ Database initialized successfully!")
    except Exception as e:
        #st.error(f"❌ Database initialization failed: {e}")
        st.stop()
    
    st.title("🔍 SalesAnalyser - AI-Powered Sales Analytics")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📊 Analysis Options")
        
        # Toggle for RAG system status
        if rag_retriever:
            st.success("✅ RAG System Active")
            st.info("Using intelligent query examples for better SQL generation")
        else:
            st.warning("⚠️ RAG System Inactive")
            st.info("Operating with fallback examples")
        
        # Show database stats
        try:
            result, _ = execute_sql_query("SELECT COUNT(*) as total_records FROM ProductSales")
            if result != "EMPTY_RESULT" and result and not str(result[0][0]).startswith("Error:"):
                st.metric("Total Records", f"{result[0][0]:,}")
        except:
            pass
        
        # Model information
        st.markdown("### 🤖 AI Model")
        st.info("**LLM**: Granite 3.3:8B\n**Embeddings**: Granite Embedding 30M")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Ask a Question About Your Sales Data")
        
        # Example questions
        with st.expander("📝 Example Questions", expanded=False):
            example_questions = [
                "Which customer has the highest total sales?",
                "What are the top 5 products by revenue in 2023?",
                "Which category generates the most profit?",
                "Show me sales by payment method",
                "Which location has the lowest sales?",
                "What is the total revenue for Electronics category?",
                "Which product has the highest average selling price?",
                "Show me monthly sales trends",
                "Which customer made the most orders?",
                "What is the profit margin for each category?"
            ]
            
            for i, example in enumerate(example_questions, 1):
                st.markdown(f"{i}. {example}")
    
    with col2:
        st.markdown("### 🚀 Quick Stats")
        # Quick database insights
        try:
            result, _ = execute_sql_query("SELECT SUM(TotalSales) FROM ProductSales")
            if result != "EMPTY_RESULT" and result and not str(result[0][0]).startswith("Error:"):
                st.metric("Total Sales", f"₹{result[0][0]:,.0f}")
            
            result, _ = execute_sql_query("SELECT COUNT(DISTINCT OrderID) FROM ProductSales")
            if result != "EMPTY_RESULT" and result and not str(result[0][0]).startswith("Error:"):
                st.metric("Total Orders", f"{result[0][0]:,}")
                
        except Exception as e:
            st.info("Loading stats...")
    
    # Question input
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Which product has the highest sales?",
        help="Ask any question about your sales data in natural language"
    )
    
    if st.button("🔍 Analyze", type="primary") or user_question:
        if user_question:
            with st.spinner("🤖 Generating SQL query..."):
                sql_query = get_sql_query(user_question)
                
                if sql_query:
                    st.markdown("### 📋 Generated SQL Query")
                    st.code(sql_query, language="sql")
                    
                    with st.spinner("⚡ Executing query..."):                        
                        dual_queries = dual_query_generator.generate_queries(user_question, sql_query) # Generate dual queries for answer and visualization                       
                        result, columns = execute_sql_query(dual_queries['answer_query']) # Execute answer query
                        
                        if result == "EMPTY_RESULT":
                            st.markdown("### 📊 Analysis Result")
                            st.info("There are no transactions/records satisfying the above question's criteria.")
                            tab1, tab2, tab3 = st.tabs(["💬 Answer", "📊 Visualization", "📋 Raw Data"])
                            
                            with tab1:
                                st.markdown("### 🎯 Analysis Result")
                                st.info("There are no transactions/records satisfying the above question's criteria.")
                            
                            with tab2:
                                st.markdown("### 📊 Data Visualization")
                                fig = create_empty_chart_with_message("No data relevant for query")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                st.markdown("### 📋 Raw Query Results")
                                st.info("No data returned from the query.")
                                
                        elif result and not (isinstance(result, list) and len(result) > 0 and str(result[0]).startswith("Error:")):                            
                            tab1, tab2, tab3 = st.tabs(["💬 Answer", "📊 Visualization", "📋 Raw Data"]) # Display results in tabs
                            
                            with tab1:
                                st.markdown("### 🎯 Analysis Result")
                                df_answer = pd.DataFrame(result, columns=columns)  # Convert to DataFrame for easier handling
                                nl_response = natural_language_interpretation(user_question, sql_query, columns, result)  # Generate natural language response                                
                                st.markdown(nl_response)
                                
                                # Show key metrics if available
                                if len(df_answer) > 0:
                                    numeric_cols = df_answer.select_dtypes(include=['int64', 'float64']).columns
                                    if len(numeric_cols) > 0:
                                        st.markdown("#### 📈 Key Metrics")
                                        metric_cols = st.columns(min(4, len(numeric_cols)))
                                        
                                        for i, col in enumerate(numeric_cols[:4]):
                                            with metric_cols[i]:
                                                if 'Sales' in col or 'Price' in col or 'Cost' in col:
                                                    st.metric(col, f"₹{df_answer[col].sum():,.0f}")
                                                else:
                                                    st.metric(col, f"{df_answer[col].sum():,.0f}")
                            
                            with tab2:
                                st.markdown("### 📊 Data Visualization")
                                viz_result, viz_columns = execute_sql_query(dual_queries['visualization_query']) # Execute visualization query
                                
                                if viz_result == "EMPTY_RESULT":
                                    fig = create_empty_chart_with_message("No data relevant for query") # Show empty chart for no data
                                    st.plotly_chart(fig, use_container_width=True)
                                elif viz_result and not str(viz_result[0]).startswith("Error:"):
                                    df_viz = pd.DataFrame(viz_result, columns=viz_columns)
                                    
                                    # Determine if chart should be created
                                    analysis = dual_queries['analysis']
                                    if should_create_enhanced_chart(df_viz, user_question, analysis):
                                        chart_config = determine_chart_data_enhanced(user_question, df_viz, viz_columns, analysis)
                                        
                                        if chart_config['x_col'] and chart_config['y_col']:
                                            create_enhanced_line_chart(df_viz, chart_config, analysis)
                                        else:
                                            fig = create_empty_chart_with_message("No data relevant for query")
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        # For direct questions, show empty chart
                                        fig = create_empty_chart_with_message("No data relevant for query")
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.info("📊 This query result is better displayed as a table")
                                        st.dataframe(df_viz, use_container_width=True)
                                else:
                                    fig = create_empty_chart_with_message("No data relevant for query")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                st.markdown("### 📋 Raw Query Results")
                                df_display = pd.DataFrame(result, columns=columns)
                                st.dataframe(df_display, use_container_width=True)
                                
                                csv = df_display.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv,
                                    file_name=f"sales_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        else:
                            st.error("❌ Query execution failed:")
                            if result:
                                st.error(result[0])
                            else:
                                st.error("Unknown error occurred")
                
                else:
                    st.error("❌ Failed to generate SQL query. Please try rephrasing your question.")
        else:
            st.warning("⚠️ Please enter a question to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **SalesAnalyser** - Powered by Granite AI Models | "
        "💡 Ask natural language questions about your sales data"
    )

if __name__ == "__main__":
    main()