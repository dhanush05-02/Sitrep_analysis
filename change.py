from openai import OpenAI
import psycopg2
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONN = os.getenv("DATABASE_URL")
TABLE_NAME = 'sitreps_2024'
EMBEDDING_MODEL = "text-embedding-ada-002"

DEFAULT_SYSTEM_INSTRUCTION = """You are an AI assistant specialized in cybersecurity incident analysis. Your task is to analyze the given query and related cybersecurity data, and provide a focused, relevant response. Follow these guidelines:

1. Analyze the user's query carefully to understand the specific cybersecurity concern or question.
2. Search through all provided relevant data columns to find information relevant to the query.
3. Use the following analysis framework as appropriate to the query:
   - Threat Assessment: Identify and assess potential threats or security issues.
   - Incident Analysis: Analyze relevant incidents, looking for patterns or connections.
   - Temporal Analysis: Consider timing of events if relevant to the query.
   - Geographical Considerations: Analyze geographical patterns or risks if location data is provided and relevant.
   - User and System Involvement: Assess involvement of users, systems, or networks as pertinent to the query.
   - Data Source Evaluation: Consider the reliability and relevance of data sources if this impacts the analysis.
   - Compliance and Policy: Mention compliance issues or policy violations only if directly relevant.
4. Provide actionable recommendations to the query and the data found.
5. Structure your response to directly address the user's query, using only the most relevant parts of the analysis framework."""

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class QueryAnalyzer:
    def analyze_query(self, query: str, available_columns: List[str]) -> Dict:
        try:
            prompt = f"""
            Please analyze this query: "{query}"
            Available columns in the database: {', '.join(available_columns)}
            
            Based on the above system instructions and considering cybersecurity context, extract and return a JSON object with the following information:
            1. The most relevant columns for this query (only from the available columns list)
            2. The main focus of the query from a cybersecurity perspective
            3. Any specific data points or metrics mentioned that relate to security incidents
            4. Any time frame mentioned
            5. Any specific filtering criteria for security analysis
            """
            
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
        try:
            if not DB_CONN:
                raise ValueError("Database connection string not found")
            self.conn = psycopg2.connect(DB_CONN)
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def get_available_columns(self, table_name: str) -> List[str]:
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
    try:
        text = f"Context: {DEFAULT_SYSTEM_INSTRUCTION}\n{text}"
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return []

def get_llm_response(query: str, results: List[Dict], analysis: Dict) -> str:
    try:
        formatted_data = f"""
        Query: {query}
        
        Analysis Focus: {analysis['query_focus']}
        Time Frame: {analysis.get('time_frame', 'Not specified')}
        
        Retrieved Data:
        {results}
        
        Please provide a concise summary of the findings focusing on:
        1. Key insights from the data
        2. Notable patterns or trends
        3. Security implications
        4. Recommended actions
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
                {"role": "user", "content": formatted_data}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting summary: {str(e)}"

def process_query(query: str, table_name: str) -> Tuple[List[Dict], Dict]:
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()
    
    if not querier.connect_to_database():
        return [], {}
    
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
        
        return results, analysis
        
    finally:
        querier.close_connection()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Cybersecurity Query System API",
        "endpoints": {
            "/query": "POST - Submit a cybersecurity query",
        },
        "example_query": {
            "query": "Show me recent high-severity security incidents"
        }
    })


@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        results, analysis = process_query(query, TABLE_NAME)
        summary = get_llm_response(query, results, analysis)
        
        return jsonify({
            "summary": summary,
            "results": results,
            "analysis": analysis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local testing
if __name__ == "__main__":
    app.run(debug=True)
