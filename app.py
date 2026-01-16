#!/usr/bin/env python3
"""
Financial Complaints Chatbot - Streamlit Interface
Task 4: Interactive complaint analysis system
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Financial Complaints Analyst",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .chat-user {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px 10px 0 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .chat-assistant {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px 10px 10px 0;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏦 Financial Complaints Analysis System</div>', unsafe_allow_html=True)
st.markdown("### Ask questions about customer complaints and analyze financial product issues")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# Load complaint data
@st.cache_data
def load_complaint_data():
    """Load complaint data from available sources"""
    try:
        # Try to load from CSV
        csv_paths = [
            "data/raw/complaints.csv",
            "data/processed/filtered_complaints.csv",
            "data/sampled/complaints_sample.csv"
        ]
        
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Loaded {len(df)} complaints from {path}")
                return df
        
        # If no CSV found, create sample data
        st.info("📊 Using sample complaint data for demonstration")
        sample_data = {
            'product': ['Credit card', 'Credit card', 'Savings account', 'Personal loan', 'Money transfer'],
            'issue': ['Unauthorized charges', 'Billing dispute', 'Unexpected fees', 'Application denial', 'Failed transfer'],
            'narrative': [
                'Customer found unauthorized transactions on monthly statement',
                'Billing error not resolved for over 60 days',
                'Monthly maintenance fee charged without notification',
                'Loan application denied despite 750 credit score',
                'International money transfer failed to reach recipient'
            ],
            'date_received': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-02-10', '2024-02-15']
        }
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_complaint_data()

# Complaint analysis function
def analyze_complaint_query(query, product_filter="All"):
    """Analyze complaint query and generate response"""
    
    query_lower = query.lower()
    analysis = ""
    sources = []
    
    # Filter data based on product
    filtered_df = df.copy()
    if product_filter != "All" and 'product' in df.columns:
        filtered_df = filtered_df[filtered_df['product'].str.contains(product_filter, case=False, na=False)]
    
    # Analyze based on query
    if not filtered_df.empty:
        total_complaints = len(filtered_df)
        
        # Product distribution
        if 'product' in filtered_df.columns:
            product_counts = filtered_df['product'].value_counts()
            top_product = product_counts.index[0] if len(product_counts) > 0 else "Unknown"
            
        # Issue analysis
        if 'issue' in filtered_df.columns:
            issue_counts = filtered_df['issue'].value_counts()
            top_issue = issue_counts.index[0] if len(issue_counts) > 0 else "Unknown"
            
        # Build analysis
        analysis = f"**Analysis Results**\n\n"
        analysis += f"📊 **Total relevant complaints:** {total_complaints}\n\n"
        
        if total_complaints > 0:
            if 'product' in filtered_df.columns:
                analysis += f"🏦 **Most affected product:** {top_product} ({product_counts.get(top_product, 0)} complaints)\n\n"
            
            if 'issue' in filtered_df.columns:
                analysis += f"⚠️ **Most common issue:** {top_issue}\n\n"
            
            # Show sample complaints
            if 'narrative' in filtered_df.columns:
                analysis += "**Sample complaints:**\n"
                for i, row in filtered_df.head(2).iterrows():
                    analysis += f"• {row.get('narrative', 'No narrative')[:100]}...\n"
        
        sources = [f"Complaint Database ({total_complaints} records)"]
        
    else:
        # Generic response if no data
        analysis = "**Financial Complaints Analysis**\n\n"
        analysis += "I can analyze complaints about:\n"
        analysis += "• **Credit Cards** - Billing issues, fraud, fees\n"
        analysis += "• **Savings Accounts** - Maintenance fees, withdrawal limits\n"
        analysis += "• **Personal Loans** - Application denials, interest rates\n"
        analysis += "• **Money Transfers** - Failed transactions, delays\n\n"
        analysis += "*Try asking about specific products or issues*"
        
        sources = ["Financial Complaints Knowledge Base"]
    
    # Add query-specific insights
    if "credit" in query_lower or "card" in query_lower:
        analysis += "\n\n💡 **Credit Card Insight:** Unauthorized charges and billing disputes are most frequent."
    elif "savings" in query_lower or "account" in query_lower:
        analysis += "\n\n💡 **Savings Account Insight:** Unexpected fees are the top complaint category."
    elif "loan" in query_lower:
        analysis += "\n\n💡 **Loan Insight:** Application processing and denial reasons are common concerns."
    elif "transfer" in query_lower or "money" in query_lower:
        analysis += "\n\n💡 **Transfer Insight:** Transaction failures and delays frequently reported."
    
    return {
        "answer": analysis,
        "sources": sources,
        "timestamp": datetime.now().isoformat(),
        "query": query
    }

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Product filter
    product_filter = st.selectbox(
        "Filter by Product Type:",
        ["All", "Credit Cards", "Savings Accounts", "Personal Loans", "Money Transfers", "Bank Accounts"]
    )
    
    st.markdown("---")
    
    # Data statistics
    st.markdown("## 📊 Data Overview")
    
    if not df.empty:
        st.metric("Total Complaints", len(df))
        
        if 'product' in df.columns:
            unique_products = df['product'].nunique()
            st.metric("Product Categories", unique_products)
    
    st.markdown("---")
    
    # Quick queries
    st.markdown("## 🚀 Quick Analysis")
    
    quick_queries = {
        "Credit Card Issues": "What are common credit card complaints?",
        "Savings Account Fees": "Tell me about savings account fee issues",
        "Loan Problems": "What issues do people have with loans?",
        "Transfer Delays": "Problems with money transfers?"
    }
    
    for label, query_text in quick_queries.items():
        if st.button(label, use_container_width=True, key=f"btn_{label}"):
            st.session_state.last_query = query_text
    
    st.markdown("---")
    
    # Session management
    st.markdown("## 📝 Session")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.rerun()
    
    # Export
    if st.session_state.chat_history:
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": st.session_state.query_count,
            "chat_history": st.session_state.chat_history
        }
        
        st.download_button(
            "💾 Export Session",
            data=json.dumps(session_data, indent=2),
            file_name=f"complaint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            use_container_width=True
        )

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    st.markdown("### 💬 Analysis Conversation")
    
    if not st.session_state.chat_history:
        st.info("""
        👋 **Welcome to the Financial Complaints Analysis System!**
        
        **How to use:**
        1. Ask questions about financial complaints in the chat below
        2. Use the sidebar to filter by product type
        3. Click quick analysis buttons for common queries
        
        **Example questions:**
        - "What are common credit card billing issues?"
        - "Tell me about savings account complaints"
        - "Problems with loan applications?"
        - "Money transfer delays and failures"
        """)
    
    # Display conversation
    for chat in st.session_state.chat_history:
        # User message
        with st.container():
            st.markdown(f'<div class="chat-user">', unsafe_allow_html=True)
            st.markdown(f"**You:** {chat.get('query', 'Unknown query')}")
            if chat.get('filter_used') and chat['filter_used'] != "All":
                st.caption(f"🔍 Filter: {chat['filter_used']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Assistant response
        if not chat.get('error', False):
            with st.container():
                st.markdown(f'<div class="chat-assistant">', unsafe_allow_html=True)
                st.markdown(chat.get('answer', 'No response available'))
                
                if chat.get('sources'):
                    with st.expander(f"📚 Sources ({len(chat['sources'])})"):
                        for source in chat['sources']:
                            st.caption(f"• {source}")
                st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Metrics dashboard
    st.markdown("### 📈 Live Metrics")
    
    with st.container():
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Questions Asked", st.session_state.query_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        # Calculate response metrics
        valid_responses = [h for h in st.session_state.chat_history if not h.get('error', False)]
        
        with st.container():
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Successful Analyses", len(valid_responses))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Product focus
    if not df.empty and 'product' in df.columns:
        product_counts = df['product'].value_counts()
        
        with st.container():
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown("**📊 Complaint Distribution**")
            for product, count in product_counts.head(3).items():
                st.caption(f"• {product}: {count}")
            st.markdown('</div>', unsafe_allow_html=True)

# Chat input at the bottom
st.markdown("---")
query_input = st.chat_input("💬 Ask about financial complaints...")

# Handle quick query buttons
if 'last_query' in st.session_state:
    query_input = st.session_state.last_query
    del st.session_state.last_query

# Process query
if query_input:
    # Add to history
    st.session_state.query_count += 1
    
    with st.spinner("🔍 Analyzing complaints..."):
        response = analyze_complaint_query(query_input, product_filter)
        
        # Add filter info
        response['filter_used'] = product_filter
        
        st.session_state.chat_history.append(response)
        
        # Rerun to update display
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "🔍 Powered by Financial Complaints Analysis • 📊 Data-driven insights • ⚖️ Consumer protection focus"
    "</div>",
    unsafe_allow_html=True
)