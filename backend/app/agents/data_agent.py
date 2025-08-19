from typing import Dict, Any, List, Optional, Union
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage
import pandas as pd
import numpy as np
import json
import io
import base64
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statistics
from scipy import stats
import warnings

from .base_agent import BaseAgent

warnings.filterwarnings('ignore')

class DataAgent(BaseAgent):
    """
    Specialized agent for data analysis, processing, and visualization
    """
    
    def __init__(self, llm_provider, websocket_manager=None):
        # Data analysis specific tools
        tools = [
            Tool(
                name="data_processor",
                description="Process and clean raw data from various formats",
                func=self._process_data
            ),
            Tool(
                name="statistical_analyzer",
                description="Perform statistical analysis on datasets",
                func=self._analyze_statistics
            ),
            Tool(
                name="data_visualizer",
                description="Create visualizations and charts from data",
                func=self._create_visualizations
            ),
            Tool(
                name="trend_analyzer",
                description="Identify trends and patterns in time-series data",
                func=self._analyze_trends
            ),
            Tool(
                name="correlation_analyzer",
                description="Analyze correlations between variables",
                func=self._analyze_correlations
            ),
            Tool(
                name="outlier_detector",
                description="Detect and handle outliers in datasets",
                func=self._detect_outliers
            ),
            Tool(
                name="data_transformer",
                description="Transform and reshape data for analysis",
                func=self._transform_data
            ),
            Tool(
                name="report_generator",
                description="Generate comprehensive data analysis reports",
                func=self._generate_report
            )
        ]
        
        super().__init__(
            name="Data Analysis Agent",
            description="Specialized agent for data processing, statistical analysis, and visualization",
            llm_provider=llm_provider,
            tools=tools,
            websocket_manager=websocket_manager
        )
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_system_prompt(self) -> str:
        return """You are a Data Analysis Agent specialized in:
        1. Processing and cleaning raw data from various formats
        2. Performing comprehensive statistical analysis
        3. Creating insightful visualizations and charts
        4. Identifying trends and patterns in data
        5. Analyzing correlations and relationships
        6. Detecting outliers and anomalies
        7. Transforming data for analysis
        8. Generating comprehensive reports
        
        Best practices you follow:
        - Always validate data quality before analysis
        - Handle missing values appropriately
        - Choose appropriate statistical methods
        - Create clear, informative visualizations
        - Provide actionable insights
        - Document assumptions and limitations
        
        Use the available tools to complete data analysis tasks effectively."""
    
    def initialize_agent(self):
        """Initialize the data analysis agent executor"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
    
    def _process_data(self, data_input: str) -> str:
        """
        Process and clean raw data from various formats
        """
        try:
            # Parse input data
            data_info = self._parse_data_input(data_input)
            
            if 'csv_data' in data_info:
                df = pd.read_csv(io.StringIO(data_info['csv_data']))
            elif 'json_data' in data_info:
                df = pd.DataFrame(data_info['json_data'])
            elif 'file_path' in data_info:
                df = self._load_data_file(data_info['file_path'])
            else:
                return "No valid data format found in input"
            
            # Perform data cleaning
            cleaned_df = self._clean_data(df)
            
            # Generate processing summary
            summary = {
                "original_shape": df.shape,
                "cleaned_shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "data_types": cleaned_df.dtypes.to_dict(),
                "missing_values": cleaned_df.isnull().sum().to_dict(),
                "summary_stats": cleaned_df.describe().to_dict(),
                "processing_steps": [
                    "Removed duplicate rows",
                    "Handled missing values",
                    "Standardized column names",
                    "Converted data types"
                ]
            }
            
            return json.dumps(summary, indent=2, default=str)
            
        except Exception as e:
            return f"Error processing data: {str(e)}"
    
    def _analyze_statistics(self, data_input: str) -> str:
        """
        Perform comprehensive statistical analysis on datasets
        """
        try:
            df = self._get_dataframe_from_input(data_input)
            if df is None:
                return "Could not load data for statistical analysis"
            
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            analysis = {
                "dataset_info": {
                    "shape": df.shape,
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols)
                },
                "descriptive_statistics": {},
                "distribution_analysis": {},
                "hypothesis_tests": {}
            }
            
            # Descriptive statistics for numeric columns
            if numeric_cols:
                desc_stats = df[numeric_cols].describe()
                analysis["descriptive_statistics"]["numeric"] = desc_stats.to_dict()
                
                # Additional statistics
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    analysis["descriptive_statistics"][f"{col}_additional"] = {
                        "skewness": stats.skew(col_data),
                        "kurtosis": stats.kurtosis(col_data),
                        "variance": col_data.var(),
                        "std_dev": col_data.std(),
                        "coefficient_of_variation": col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                    }
            
            # Categorical analysis
            if categorical_cols:
                cat_analysis = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    cat_analysis[col] = {
                        "unique_values": len(value_counts),
                        "most_common": value_counts.head(5).to_dict(),
                        "missing_count": df[col].isnull().sum()
                    }
                analysis["descriptive_statistics"]["categorical"] = cat_analysis
            
            # Distribution tests for numeric columns
            if numeric_cols:
                dist_tests = {}
                for col in numeric_cols[:5]:  # Limit to first 5 columns for performance
                    col_data = df[col].dropna()
                    if len(col_data) > 3:
                        # Normality test
                        shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000])  # Limit sample size
                        dist_tests[col] = {
                            "normality_test": {
                                "statistic": shapiro_stat,
                                "p_value": shapiro_p,
                                "is_normal": shapiro_p > 0.05
                            }
                        }
                analysis["distribution_analysis"] = dist_tests
            
            return json.dumps(analysis, indent=2, default=str)
            
        except Exception as e:
            return f"Error in statistical analysis: {str(e)}"
    
    def _create_visualizations(self, data_input: str) -> str:
        """
        Create visualizations and charts from data
        """
        try:
            request = self._parse_visualization_request(data_input)
            df = self._get_dataframe_from_input(request.get('data', data_input))
            
            if df is None:
                return "Could not load data for visualization"
            
            chart_type = request.get('chart_type', 'auto')
            columns = request.get('columns', [])
            
            visualizations = []
            
            # Auto-generate visualizations based on data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if chart_type == 'auto' or chart_type == 'distribution':
                # Distribution plots for numeric columns
                for col in numeric_cols[:4]:  # Limit to 4 columns
                    plt.figure(figsize=(10, 6))
                    plt.subplot(2, 2, 1)
                    df[col].hist(bins=30, alpha=0.7)
                    plt.title(f'Distribution of {col}')
                    
                    plt.subplot(2, 2, 2)
                    df[col].plot(kind='box')
                    plt.title(f'Box Plot of {col}')
                    
                    chart_path = self._save_chart(f'distribution_{col}')
                    visualizations.append({
                        "type": "distribution",
                        "column": col,
                        "path": chart_path
                    })
                    plt.close()
            
            if chart_type == 'auto' or chart_type == 'correlation':
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(12, 8))
                    corr_matrix = df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title('Correlation Heatmap')
                    chart_path = self._save_chart('correlation_heatmap')
                    visualizations.append({
                        "type": "correlation",
                        "path": chart_path
                    })
                    plt.close()
            
            if chart_type == 'auto' or chart_type == 'categorical':
                # Categorical plots
                for col in categorical_cols[:3]:  # Limit to 3 columns
                    plt.figure(figsize=(12, 6))
                    value_counts = df[col].value_counts().head(10)
                    value_counts.plot(kind='bar')
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    chart_path = self._save_chart(f'categorical_{col}')
                    visualizations.append({
                        "type": "categorical",
                        "column": col,
                        "path": chart_path
                    })
                    plt.close()
            
            result = {
                "visualizations_created": len(visualizations),
                "charts": visualizations,
                "summary": f"Generated {len(visualizations)} visualizations"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error creating visualizations: {str(e)}"
    
    def _analyze_trends(self, data_input: str) -> str:
        """
        Identify trends and patterns in time-series data
        """
        try:
            df = self._get_dataframe_from_input(data_input)
            if df is None:
                return "Could not load data for trend analysis"
            
            # Try to identify date/time columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        continue
            
            if not date_cols:
                return "No date/time columns found for trend analysis"
            
            # Use the first date column
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Analyze trends for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            trends = {}
            for col in numeric_cols[:5]:  # Limit to 5 columns
                # Calculate trend metrics
                values = df[col].dropna()
                if len(values) > 2:
                    # Linear regression for trend
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends[col] = {
                        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "is_significant": p_value < 0.05,
                        "start_value": values.iloc[0],
                        "end_value": values.iloc[-1],
                        "percent_change": ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100 if values.iloc[0] != 0 else 0
                    }
            
            # Create trend visualization
            if numeric_cols:
                plt.figure(figsize=(15, 8))
                for i, col in enumerate(numeric_cols[:4]):
                    plt.subplot(2, 2, i+1)
                    plt.plot(df[date_col], df[col])
                    plt.title(f'Trend: {col}')
                    plt.xticks(rotation=45)
                
                chart_path = self._save_chart('trends_analysis')
                
            result = {
                "date_column": date_col,
                "analyzed_columns": list(trends.keys()),
                "trends": trends,
                "visualization": chart_path if 'chart_path' in locals() else None
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error in trend analysis: {str(e)}"
    
    def _analyze_correlations(self, data_input: str) -> str:
        """
        Analyze correlations between variables
        """
        try:
            df = self._get_dataframe_from_input(data_input)
            if df is None:
                return "Could not load data for correlation analysis"
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return "Need at least 2 numeric columns for correlation analysis"
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations (absolute value > 0.7)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            "variable_1": col1,
                            "variable_2": col2,
                            "correlation": corr_value,
                            "strength": "very strong" if abs(corr_value) > 0.9 else "strong"
                        })
            
            # Perform correlation significance tests
            correlation_tests = {}
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    data1 = df[col1].dropna()
                    data2 = df[col2].dropna()
                    
                    # Find common indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 3:
                        corr_coef, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                        correlation_tests[f"{col1}_vs_{col2}"] = {
                            "pearson_r": corr_coef,
                            "p_value": p_value,
                            "is_significant": p_value < 0.05
                        }
            
            result = {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": strong_correlations,
                "correlation_tests": correlation_tests,
                "summary": {
                    "total_pairs": len(correlation_tests),
                    "significant_correlations": sum(1 for test in correlation_tests.values() if test['is_significant']),
                    "strong_correlations_count": len(strong_correlations)
                }
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error in correlation analysis: {str(e)}"
    
    def _detect_outliers(self, data_input: str) -> str:
        """
        Detect and handle outliers in datasets
        """
        try:
            df = self._get_dataframe_from_input(data_input)
            if df is None:
                return "Could not load data for outlier detection"
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outlier_results = {}
            
            for col in numeric_cols:
                data = df[col].dropna()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                zscore_outliers = data[z_scores > 3]
                
                # Modified Z-score method (using median)
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                modified_zscore_outliers = data[np.abs(modified_z_scores) > 3.5]
                
                outlier_results[col] = {
                    "iqr_method": {
                        "outliers_count": len(iqr_outliers),
                        "outliers_percentage": (len(iqr_outliers) / len(data)) * 100,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "outlier_values": iqr_outliers.tolist()[:10]  # Limit to 10 values
                    },
                    "zscore_method": {
                        "outliers_count": len(zscore_outliers),
                        "outliers_percentage": (len(zscore_outliers) / len(data)) * 100,
                        "outlier_values": zscore_outliers.tolist()[:10]
                    },
                    "modified_zscore_method": {
                        "outliers_count": len(modified_zscore_outliers),
                        "outliers_percentage": (len(modified_zscore_outliers) / len(data)) * 100,
                        "outlier_values": modified_zscore_outliers.tolist()[:10]
                    }
                }
            
            # Create outlier visualization
            if numeric_cols:
                fig, axes = plt.subplots(2, min(2, len(numeric_cols)), figsize=(15, 10))
                if len(numeric_cols) == 1:
                    axes = [axes]
                elif len(numeric_cols) > 1:
                    axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols[:4]):
                    if i < len(axes):
                        axes[i].boxplot(df[col].dropna())
                        axes[i].set_title(f'Outliers in {col}')
                
                chart_path = self._save_chart('outliers_boxplots')
            
            result = {
                "columns_analyzed": len(outlier_results),
                "outlier_analysis": outlier_results,
                "visualization": chart_path if 'chart_path' in locals() else None,
                "recommendations": [
                    "Review outliers to determine if they are data errors or genuine extreme values",
                    "Consider the impact of outliers on your analysis",
                    "Apply appropriate outlier treatment based on your domain knowledge"
                ]
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error in outlier detection: {str(e)}"
    
    def _transform_data(self, data_input: str) -> str:
        """
        Transform and reshape data for analysis
        """
        try:
            request = self._parse_transform_request(data_input)
            df = self._get_dataframe_from_input(request.get('data', data_input))
            
            if df is None:
                return "Could not load data for transformation"
            
            transformation_type = request.get('transformation', 'normalize')
            columns = request.get('columns', [])
            
            transformed_data = df.copy()
            transformations_applied = []
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = columns if columns else numeric_cols
            
            if transformation_type in ['normalize', 'standardize']:
                # Standardization (z-score normalization)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                transformed_data[target_cols] = scaler.fit_transform(df[target_cols])
                transformations_applied.append(f"Standardized columns: {target_cols}")
            
            elif transformation_type == 'minmax':
                # Min-Max scaling
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                transformed_data[target_cols] = scaler.fit_transform(df[target_cols])
                transformations_applied.append(f"Min-Max scaled columns: {target_cols}")
            
            elif transformation_type == 'log':
                # Log transformation
                for col in target_cols:
                    if (df[col] > 0).all():
                        transformed_data[col] = np.log(df[col])
                        transformations_applied.append(f"Log transformed: {col}")
                    else:
                        transformations_applied.append(f"Skipped {col} (contains non-positive values)")
            
            elif transformation_type == 'sqrt':
                # Square root transformation
                for col in target_cols:
                    if (df[col] >= 0).all():
                        transformed_data[col] = np.sqrt(df[col])
                        transformations_applied.append(f"Square root transformed: {col}")
                    else:
                        transformations_applied.append(f"Skipped {col} (contains negative values)")
            
            # Calculate transformation summary
            summary = {
                "original_shape": df.shape,
                "transformed_shape": transformed_data.shape,
                "transformation_type": transformation_type,
                "transformations_applied": transformations_applied,
                "before_after_stats": {}
            }
            
            # Compare before and after statistics
            for col in target_cols:
                if col in df.columns and col in transformed_data.columns:
                    summary["before_after_stats"][col] = {
                        "original": {
                            "mean": df[col].mean(),
                            "std": df[col].std(),
                            "min": df[col].min(),
                            "max": df[col].max()
                        },
                        "transformed": {
                            "mean": transformed_data[col].mean(),
                            "std": transformed_data[col].std(),
                            "min": transformed_data[col].min(),
                            "max": transformed_data[col].max()
                        }
                    }
            
            return json.dumps(summary, indent=2, default=str)
            
        except Exception as e:
            return f"Error in data transformation: {str(e)}"
    
    def _generate_report(self, data_input: str) -> str:
        """
        Generate comprehensive data analysis reports
        """
        try:
            df = self._get_dataframe_from_input(data_input)
            if df is None:
                return "Could not load data for report generation"
            
            # Generate comprehensive report
            report = {
                "executive_summary": self._generate_executive_summary(df),
                "data_overview": self._generate_data_overview(df),
                "key_insights": self._generate_key_insights(df),
                "detailed_analysis": self._generate_detailed_analysis(df),
                "recommendations": self._generate_recommendations_from_analysis(df),
                "technical_details": self._generate_technical_details(df)
            }
            
            # Format as readable report
            formatted_report = self._format_analysis_report(report)
            
            return formatted_report
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    # Helper methods
    def _parse_data_input(self, data_input: str) -> Dict[str, Any]:
        """Parse various data input formats"""
        try:
            return json.loads(data_input)
        except:
            return {"raw_data": data_input}
    
    def _parse_visualization_request(self, data_input: str) -> Dict[str, Any]:
        """Parse visualization request"""
        try:
            return json.loads(data_input)
        except:
            return {"data": data_input, "chart_type": "auto"}
    
    def _parse_transform_request(self, data_input: str) -> Dict[str, Any]:
        """Parse transformation request"""
        try:
            return json.loads(data_input)
        except:
            return {"data": data_input, "transformation": "normalize"}
    
    def _get_dataframe_from_input(self, data_input: str) -> Optional[pd.DataFrame]:
        """Convert input to DataFrame"""
        try:
            data_info = self._parse_data_input(data_input)
            
            if 'csv_data' in data_info:
                return pd.read_csv(io.StringIO(data_info['csv_data']))
            elif 'json_data' in data_info:
                return pd.DataFrame(data_info['json_data'])
            elif 'file_path' in data_info:
                return self._load_data_file(data_info['file_path'])
            else:
                # Try to create DataFrame from raw data
                try:
                    return pd.read_csv(io.StringIO(data_input))
                except:
                    return None
        except:
            return None
    
    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown')
        
        # Standardize column names
        cleaned_df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in cleaned_df.columns]
        
        return cleaned_df
    
    def _save_chart(self, chart_name: str) -> str:
        """Save chart to temporary file and return path"""
        try:
            temp_dir = tempfile.gettempdir()
            chart_path = os.path.join(temp_dir, f"{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            return chart_path
        except Exception as e:
            return f"Error saving chart: {str(e)}"
    
    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary of the data"""
        return f"""
        Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
        {len(df.select_dtypes(include=[np.number]).columns)} numeric columns and {len(df.select_dtypes(include=['object', 'category']).columns)} categorical columns.
        Data quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}% complete.
        """
    
    def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data overview"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    
    def _generate_key_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate key insights from data"""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Find column with highest variation
            cv_values = {col: df[col].std() / df[col].mean() if df[col].mean() != 0 else 0 for col in numeric_cols}
            highest_cv_col = max(cv_values, key=cv_values.get)
            insights.append(f"'{highest_cv_col}' shows the highest variation (CV: {cv_values[highest_cv_col]:.2f})")
        
        # Check for potential data quality issues
        if df.duplicated().sum() > 0:
            insights.append(f"Dataset contains {df.duplicated().sum()} duplicate rows")
        
        return insights
    
    def _generate_detailed_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            "descriptive_statistics": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            "data_quality": {
                "completeness": ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100),
                "duplicates": df.duplicated().sum(),
                "unique_values_per_column": df.nunique().to_dict()
            }
        }
    
    def _generate_recommendations_from_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if df.isnull().sum().sum() > 0:
            recommendations.append("Address missing values before further analysis")
        
        if df.duplicated().sum() > 0:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            recommendations.append("Perform correlation analysis to identify relationships")
        
        return recommendations
    
    def _generate_technical_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical details"""
        return {
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "data_types_count": df.dtypes.value_counts().to_dict(),
            "index_type": str(type(df.index)),
            "column_names": list(df.columns)
        }
    
    def _format_analysis_report(self, report: Dict[str, Any]) -> str:
        """Format analysis report as readable text"""
        formatted = f"""
# Data Analysis Report

## Executive Summary
{report['executive_summary']}

## Key Insights
"""
        for insight in report['key_insights']:
            formatted += f"- {insight}\n"
        
        formatted += f"""
## Data Overview
- Shape: {report['data_overview']['shape']}
- Columns: {len(report['data_overview']['columns'])}
- Memory Usage: {report['technical_details']['memory_usage_mb']:.2f} MB

## Recommendations
"""
        for rec in report['recommendations']:
            formatted += f"- {rec}\n"
        
        return formatted