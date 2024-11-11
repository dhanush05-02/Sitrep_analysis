from openai import OpenAI
import psycopg2
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONN = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not DB_CONN:
    raise ValueError("Missing required environment variables. Please check your .env file.")

TABLE_NAME = 'sitreps_2024'
EMBEDDING_MODEL = "text-embedding-ada-002"

DEFAULT_SYSTEM_INSTRUCTION = """You are an AI assistant specialized in cybersecurity incident analysis..."""  # Keep the original instruction

class QueryAnalyzer:
    def analyze_query(self, query: str, available_columns: List[str]) -> Dict:
        """Analyze the user query to determine relevant columns and query intention"""
        try:
            prompt = f"""
{DEFAULT_SYSTEM_INSTRUCTION}

Please analyze this query: "{query}"

Available columns in the database: {', '.join(available_columns)}

Based on the above system instructions and considering cybersecurity context, extract and return a JSON object with the following information:
1. The most relevant columns for this query (only from the available columns list)
2. The main focus of the query from a cybersecurity perspective
3. Any specific data points or metrics mentioned that relate to security incidents
4. Any time frame mentioned
5. Any specific filtering criteria for security analysis

Format the response as a JSON object with these exact keys:
{{
    "relevant_columns": [],
    "query_focus": "",
    "specific_data_points": [],
    "time_frame": "",
    "filter_criteria": []
}}
"""
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            return eval(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            return {
                "relevant_columns": [],
                "query_focus": "",
                "specific_data_points": [],
                "time_frame": "",
                "filter_criteria": []
            }

class DatabaseQuerier:
    def __init__(self):
        self.conn = None
        self.available_columns = []

    def connect_to_database(self):
        """Create connection to database"""
        try:
            if not DB_CONN:
                raise ValueError("Database connection string not found")
            self.conn = psycopg2.connect(DB_CONN)
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_available_columns(self, table_name: str) -> List[str]:
        """Get list of available columns from the specified table"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
                
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table_name,))
                self.available_columns = [row[0] for row in cur.fetchall()]
                return self.available_columns
        except Exception as e:
            print(f"Error fetching columns: {str(e)}")
            return []

    def search_similar_records(self, query_embedding: List[float], relevant_columns: List[str], 
                             table_name: str, limit: int = 5) -> List[Dict]:
        """Search for similar records based on embedding"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                columns_str = ", ".join(relevant_columns) if relevant_columns else "*"
                
                cur.execute(f"""
                    SELECT {columns_str}
                    FROM {table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, limit))
                
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                
                return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            print(f"Error searching records: {str(e)}")
            return []

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's embedding model"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return []

def get_llm_response(query: str, formatted_data: str) -> str:
    """Get response from OpenAI based on the query and formatted data"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"""
Based on this query: "{query}"
And these results: {formatted_data}

Please provide a concise summary of the findings in a clear, professional manner.
Focus on the key security insights and relevant details from the data.
"""}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def process_query(query: str, table_name: str) -> Tuple[List[Dict], Dict, str]:
    """Process a natural language query and return relevant data"""
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()
    
    if not querier.connect_to_database():
        return [], {}, "Error connecting to database"
    
    try:
        available_columns = querier.get_available_columns(table_name)
        analysis = analyzer.analyze_query(query, available_columns)
        
        query_with_context = f"""
Context: {DEFAULT_SYSTEM_INSTRUCTION}
Query: {query}
Analysis Focus: {analysis['query_focus']}
"""
        query_embedding = get_embedding(query_with_context)
        
        results = querier.search_similar_records(
            query_embedding,
            analysis['relevant_columns'],
            table_name
        )
        
        # Generate summary from results
        summary = get_llm_response(query, str(results))
        
        return results, analysis, summary
        
    finally:
        querier.close_connection()

app = Flask(__name__)

@app.route('/')
def home():
    return "Security Analysis API is running"

@app.route('/analyze', methods=['POST'])
def analyze_security_query():
    """
    API endpoint to analyze security queries
    
    Expected JSON input:
    {
        "query": "your security question here"
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body'
            }), 400

        query = data['query']
        results, analysis, summary = process_query(query, TABLE_NAME)

        return jsonify({
            'query': query,
            'analysis': analysis,
            'results': results,
            'summary': summary
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run
