from typing import Dict, Any, List, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage
import json
import re
from datetime import datetime
from textblob import TextBlob
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import markdown
import html2text

from .base_agent import BaseAgent

class CommunicationAgent(BaseAgent):
    """
    Specialized agent for communication tasks including email drafting, reports, 
    sentiment analysis, and multi-language support
    """
    
    def __init__(self, llm_provider, websocket_manager=None):
        # Communication-specific tools
        tools = [
            Tool(
                name="email_drafter",
                description="Draft professional emails based on context and requirements",
                func=self._draft_email
            ),
            Tool(
                name="report_generator",
                description="Generate comprehensive reports from data and analysis",
                func=self._generate_report
            ),
            Tool(
                name="sentiment_analyzer",
                description="Analyze sentiment of text content",
                func=self._analyze_sentiment
            ),
            Tool(
                name="content_summarizer",
                description="Summarize long content into key points",
                func=self._summarize_content
            ),
            Tool(
                name="language_translator",
                description="Translate text between different languages",
                func=self._translate_text
            ),
            Tool(
                name="document_formatter",
                description="Format documents in various formats (HTML, Markdown, PDF)",
                func=self._format_document
            ),
            Tool(
                name="tone_analyzer",
                description="Analyze and adjust tone of written content",
                func=self._analyze_tone
            ),
            Tool(
                name="presentation_creator",
                description="Create presentation outlines and content",
                func=self._create_presentation
            ),
            Tool(
                name="social_media_creator",
                description="Create social media content and posts",
                func=self._create_social_content
            )
        ]
        
        super().__init__(
            name="Communication Agent",
            description="Specialized agent for drafting emails, reports, sentiment analysis, and multi-language communication",
            llm_provider=llm_provider,
            tools=tools,
            websocket_manager=websocket_manager
        )
    
    def get_system_prompt(self) -> str:
        return """You are a Communication Agent specialized in:
        1. Drafting professional emails and messages
        2. Creating comprehensive reports and documents
        3. Analyzing sentiment and tone of content
        4. Summarizing complex information
        5. Multi-language translation and localization
        6. Document formatting and presentation
        7. Social media content creation
        8. Professional communication optimization
        
        Communication principles you follow:
        - Clarity and conciseness
        - Appropriate tone for audience
        - Professional formatting
        - Cultural sensitivity
        - Accessibility considerations
        - Brand consistency
        - Effective call-to-actions
        
        Use the available tools to complete communication tasks effectively."""
    
    def initialize_agent(self):
        """Initialize the communication agent executor"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
    
    def _draft_email(self, request_input: str) -> str:
        """
        Draft professional emails based on context and requirements
        """
        try:
            request = self._parse_email_request(request_input)
            
            email_type = request.get('type', 'general')
            recipient = request.get('recipient', 'Recipient')
            subject_context = request.get('subject', 'General Communication')
            content_context = request.get('content', '')
            tone = request.get('tone', 'professional')
            urgency = request.get('urgency', 'normal')
            
            # Generate email based on type
            if email_type == 'follow_up':
                email_content = self._draft_followup_email(recipient, subject_context, content_context, tone)
            elif email_type == 'meeting_request':
                email_content = self._draft_meeting_request(recipient, subject_context, content_context, tone)
            elif email_type == 'project_update':
                email_content = self._draft_project_update(recipient, subject_context, content_context, tone)
            elif email_type == 'client_proposal':
                email_content = self._draft_client_proposal(recipient, subject_context, content_context, tone)
            elif email_type == 'complaint_response':
                email_content = self._draft_complaint_response(recipient, subject_context, content_context, tone)
            else:
                email_content = self._draft_general_email(recipient, subject_context, content_context, tone)
            
            # Add urgency markers if needed
            if urgency == 'high':
                email_content['subject'] = f"URGENT: {email_content['subject']}"
            elif urgency == 'low':
                email_content['subject'] = f"FYI: {email_content['subject']}"
            
            return json.dumps(email_content, indent=2)
            
        except Exception as e:
            return f"Error drafting email: {str(e)}"
    
    def _generate_report(self, report_request: str) -> str:
        """
        Generate comprehensive reports from data and analysis
        """
        try:
            request = self._parse_report_request(report_request)
            
            report_type = request.get('type', 'general')
            data = request.get('data', {})
            audience = request.get('audience', 'general')
            format_type = request.get('format', 'markdown')
            
            if report_type == 'executive_summary':
                report = self._create_executive_summary(data, audience)
            elif report_type == 'technical_analysis':
                report = self._create_technical_analysis(data, audience)
            elif report_type == 'market_research':
                report = self._create_market_research_report(data, audience)
            elif report_type == 'project_status':
                report = self._create_project_status_report(data, audience)
            elif report_type == 'performance_analysis':
                report = self._create_performance_analysis(data, audience)
            else:
                report = self._create_general_report(data, audience)
            
            # Format the report
            if format_type == 'html':
                formatted_report = markdown.markdown(report)
            elif format_type == 'text':
                formatted_report = html2text.html2text(markdown.markdown(report))
            else:
                formatted_report = report
            
            result = {
                "report_type": report_type,
                "audience": audience,
                "format": format_type,
                "content": formatted_report,
                "word_count": len(formatted_report.split()),
                "generated_at": datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def _analyze_sentiment(self, text_input: str) -> str:
        """
        Analyze sentiment of text content
        """
        try:
            request = self._parse_text_input(text_input)
            text = request.get('text', text_input)
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            
            # Get sentiment polarity and subjectivity
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = "Positive"
            elif polarity < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            # Classify subjectivity
            if subjectivity > 0.5:
                subjectivity_label = "Subjective"
            else:
                subjectivity_label = "Objective"
            
            # Additional analysis
            word_count = len(text.split())
            sentence_count = len(blob.sentences)
            
            # Emotion detection (basic)
            emotions = self._detect_emotions(text)
            
            result = {
                "sentiment": {
                    "label": sentiment_label,
                    "polarity": polarity,
                    "confidence": abs(polarity)
                },
                "subjectivity": {
                    "label": subjectivity_label,
                    "score": subjectivity
                },
                "text_stats": {
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0
                },
                "emotions": emotions,
                "analysis_summary": f"The text has a {sentiment_label.lower()} sentiment with {subjectivity_label.lower()} tone."
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"
    
    def _summarize_content(self, content_input: str) -> str:
        """
        Summarize long content into key points
        """
        try:
            request = self._parse_content_request(content_input)
            content = request.get('content', content_input)
            summary_type = request.get('type', 'bullet_points')
            max_length = request.get('max_length', 200)
            
            # Basic text processing
            sentences = self._extract_sentences(content)
            
            if summary_type == 'bullet_points':
                summary = self._create_bullet_summary(sentences, max_length)
            elif summary_type == 'paragraph':
                summary = self._create_paragraph_summary(sentences, max_length)
            elif summary_type == 'abstract':
                summary = self._create_abstract_summary(sentences, max_length)
            else:
                summary = self._create_key_points_summary(sentences, max_length)
            
            result = {
                "original_length": len(content.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": len(summary.split()) / len(content.split()),
                "summary_type": summary_type,
                "summary": summary,
                "key_topics": self._extract_key_topics(content)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error summarizing content: {str(e)}"
    
    def _translate_text(self, translation_request: str) -> str:
        """
        Translate text between different languages
        """
        try:
            request = self._parse_translation_request(translation_request)
            text = request.get('text', '')
            source_lang = request.get('source_language', 'auto')
            target_lang = request.get('target_language', 'en')
            
            # Note: This is a placeholder for translation functionality
            # In a real implementation, you would use a translation service like Google Translate API
            
            result = {
                "original_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "translated_text": f"[Translation to {target_lang}]: {text}",
                "confidence": 0.95,
                "note": "Translation service integration required for actual translation"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error translating text: {str(e)}"
    
    def _format_document(self, format_request: str) -> str:
        """
        Format documents in various formats (HTML, Markdown, PDF)
        """
        try:
            request = self._parse_format_request(format_request)
            content = request.get('content', '')
            source_format = request.get('source_format', 'text')
            target_format = request.get('target_format', 'markdown')
            
            if target_format == 'html' and source_format == 'markdown':
                formatted_content = markdown.markdown(content)
            elif target_format == 'text' and source_format == 'html':
                formatted_content = html2text.html2text(content)
            elif target_format == 'markdown' and source_format == 'text':
                formatted_content = self._text_to_markdown(content)
            else:
                formatted_content = content
            
            result = {
                "source_format": source_format,
                "target_format": target_format,
                "formatted_content": formatted_content,
                "formatting_applied": f"Converted from {source_format} to {target_format}"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error formatting document: {str(e)}"
    
    def _analyze_tone(self, tone_request: str) -> str:
        """
        Analyze and adjust tone of written content
        """
        try:
            request = self._parse_tone_request(tone_request)
            text = request.get('text', '')
            desired_tone = request.get('desired_tone', 'professional')
            
            # Analyze current tone
            current_tone = self._detect_tone(text)
            
            # Suggest adjustments
            adjustments = self._suggest_tone_adjustments(text, current_tone, desired_tone)
            
            # Generate adjusted version
            adjusted_text = self._adjust_text_tone(text, desired_tone)
            
            result = {
                "original_text": text,
                "current_tone": current_tone,
                "desired_tone": desired_tone,
                "tone_analysis": {
                    "formality": self._analyze_formality(text),
                    "emotional_intensity": self._analyze_emotional_intensity(text),
                    "confidence_level": self._analyze_confidence(text)
                },
                "adjustments_suggested": adjustments,
                "adjusted_text": adjusted_text
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error analyzing tone: {str(e)}"
    
    def _create_presentation(self, presentation_request: str) -> str:
        """
        Create presentation outlines and content
        """
        try:
            request = self._parse_presentation_request(presentation_request)
            topic = request.get('topic', 'General Presentation')
            audience = request.get('audience', 'general')
            duration = request.get('duration', 15)  # minutes
            content_data = request.get('data', {})
            
            # Calculate slide count (roughly 1 slide per minute)
            slide_count = max(5, min(duration, 20))
            
            presentation = {
                "title": topic,
                "audience": audience,
                "duration_minutes": duration,
                "slide_count": slide_count,
                "outline": self._create_presentation_outline(topic, slide_count),
                "slides": self._create_presentation_slides(topic, audience, slide_count, content_data)
            }
            
            return json.dumps(presentation, indent=2)
            
        except Exception as e:
            return f"Error creating presentation: {str(e)}"
    
    def _create_social_content(self, social_request: str) -> str:
        """
        Create social media content and posts
        """
        try:
            request = self._parse_social_request(social_request)
            platform = request.get('platform', 'general')
            content_type = request.get('type', 'post')
            topic = request.get('topic', '')
            tone = request.get('tone', 'engaging')
            
            if platform == 'twitter':
                content = self._create_twitter_content(topic, content_type, tone)
            elif platform == 'linkedin':
                content = self._create_linkedin_content(topic, content_type, tone)
            elif platform == 'facebook':
                content = self._create_facebook_content(topic, content_type, tone)
            elif platform == 'instagram':
                content = self._create_instagram_content(topic, content_type, tone)
            else:
                content = self._create_general_social_content(topic, content_type, tone)
            
            result = {
                "platform": platform,
                "content_type": content_type,
                "topic": topic,
                "tone": tone,
                "content": content,
                "hashtags": self._generate_hashtags(topic, platform),
                "best_posting_time": self._suggest_posting_time(platform),
                "engagement_tips": self._get_engagement_tips(platform)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error creating social content: {str(e)}"
    
    # Helper methods for email drafting
    def _parse_email_request(self, request: str) -> Dict[str, Any]:
        """Parse email request"""
        try:
            return json.loads(request)
        except:
            return {"content": request, "type": "general"}
    
    def _draft_general_email(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a general email"""
        return {
            "to": recipient,
            "subject": subject,
            "body": f"Dear {recipient},\n\n{content}\n\nBest regards,\n[Your Name]",
            "tone": tone
        }
    
    def _draft_followup_email(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a follow-up email"""
        return {
            "to": recipient,
            "subject": f"Follow-up: {subject}",
            "body": f"Dear {recipient},\n\nI hope this email finds you well. I wanted to follow up on our previous conversation regarding {content}.\n\nI would appreciate your feedback at your earliest convenience.\n\nThank you for your time.\n\nBest regards,\n[Your Name]",
            "tone": tone
        }
    
    def _draft_meeting_request(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a meeting request email"""
        return {
            "to": recipient,
            "subject": f"Meeting Request: {subject}",
            "body": f"Dear {recipient},\n\nI hope you are doing well. I would like to schedule a meeting to discuss {content}.\n\nPlease let me know your availability for the coming week, and I will send a calendar invitation accordingly.\n\nThank you for your time.\n\nBest regards,\n[Your Name]",
            "tone": tone
        }
    
    def _draft_project_update(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a project update email"""
        return {
            "to": recipient,
            "subject": f"Project Update: {subject}",
            "body": f"Dear {recipient},\n\nI wanted to provide you with an update on {subject}.\n\n{content}\n\nPlease let me know if you have any questions or need additional information.\n\nBest regards,\n[Your Name]",
            "tone": tone
        }
    
    def _draft_client_proposal(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a client proposal email"""
        return {
            "to": recipient,
            "subject": f"Proposal: {subject}",
            "body": f"Dear {recipient},\n\nThank you for your interest in our services. I am pleased to present our proposal for {subject}.\n\n{content}\n\nWe look forward to the opportunity to work with you. Please let me know if you have any questions.\n\nBest regards,\n[Your Name]",
            "tone": tone
        }
    
    def _draft_complaint_response(self, recipient: str, subject: str, content: str, tone: str) -> Dict[str, str]:
        """Draft a complaint response email"""
        return {
            "to": recipient,
            "subject": f"Re: {subject}",
            "body": f"Dear {recipient},\n\nThank you for bringing this matter to our attention. I sincerely apologize for any inconvenience you have experienced.\n\n{content}\n\nWe value your feedback and are committed to improving our service.\n\nSincerely,\n[Your Name]",
            "tone": "empathetic"
        }
    
    # Helper methods for report generation
    def _parse_report_request(self, request: str) -> Dict[str, Any]:
        """Parse report request"""
        try:
            return json.loads(request)
        except:
            return {"data": request, "type": "general"}
    
    def _create_executive_summary(self, data: Dict[str, Any], audience: str) -> str:
        """Create an executive summary report"""
        return f"""# Executive Summary
        
## Overview
This report provides a high-level overview of key findings and recommendations.

## Key Points
- Strategic implications have been identified
- Performance metrics show positive trends
- Actionable recommendations are provided

## Next Steps
1. Review detailed findings
2. Implement priority recommendations  
3. Monitor progress metrics

*Report generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    def _create_technical_analysis(self, data: Dict[str, Any], audience: str) -> str:
        """Create a technical analysis report"""
        return f"""# Technical Analysis Report
        
## Methodology
Comprehensive analysis was performed using industry-standard techniques.

## Findings
- Technical specifications have been evaluated
- Performance benchmarks established
- Risk factors identified

## Recommendations
1. Implement technical improvements
2. Address identified vulnerabilities
3. Establish monitoring protocols

*Technical analysis completed on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    def _create_market_research_report(self, data: Dict[str, Any], audience: str) -> str:
        """Create a market research report"""
        return f"""# Market Research Report
        
## Market Overview
Current market conditions and trends analysis.

## Competitive Landscape
- Key competitors identified
- Market positioning analyzed
- Opportunities assessed

## Recommendations
1. Strategic market positioning
2. Competitive differentiation
3. Growth opportunity pursuit

*Market research conducted on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    def _create_project_status_report(self, data: Dict[str, Any], audience: str) -> str:
        """Create a project status report"""
        return f"""# Project Status Report
        
## Current Status
Project is progressing according to plan with minor adjustments.

## Milestones
- ✅ Phase 1 completed
- 🔄 Phase 2 in progress
- ⏳ Phase 3 planned

## Issues & Risks
No critical issues identified at this time.

## Next Steps
1. Complete current phase activities
2. Begin preparation for next phase
3. Stakeholder review meeting

*Status updated on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    def _create_performance_analysis(self, data: Dict[str, Any], audience: str) -> str:
        """Create a performance analysis report"""
        return f"""# Performance Analysis Report
        
## Performance Overview
System performance has been evaluated across multiple metrics.

## Key Metrics
- Response time: Acceptable
- Throughput: Within targets
- Error rate: Below threshold

## Analysis
Performance indicators show stable operation with room for optimization.

## Recommendations
1. Optimize bottleneck areas
2. Implement monitoring enhancements
3. Plan capacity improvements

*Analysis performed on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    def _create_general_report(self, data: Dict[str, Any], audience: str) -> str:
        """Create a general report"""
        return f"""# General Report
        
## Summary
This report provides an analysis of the provided data and findings.

## Key Findings
- Data has been thoroughly analyzed
- Patterns and trends identified
- Actionable insights developed

## Conclusions
Based on the analysis, several recommendations have been formulated.

## Recommendations
1. Address priority areas
2. Implement suggested improvements
3. Monitor progress regularly

*Report generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    # Additional helper methods
    def _parse_text_input(self, text_input: str) -> Dict[str, Any]:
        """Parse text input for analysis"""
        try:
            return json.loads(text_input)
        except:
            return {"text": text_input}
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Basic emotion detection"""
        emotions = []
        
        # Simple keyword-based emotion detection
        if any(word in text.lower() for word in ['happy', 'joy', 'excited', 'pleased']):
            emotions.append('joy')
        if any(word in text.lower() for word in ['sad', 'disappointed', 'upset']):
            emotions.append('sadness')
        if any(word in text.lower() for word in ['angry', 'frustrated', 'annoyed']):
            emotions.append('anger')
        if any(word in text.lower() for word in ['afraid', 'scared', 'worried']):
            emotions.append('fear')
        
        return emotions if emotions else ['neutral']
    
    def _parse_content_request(self, content_input: str) -> Dict[str, Any]:
        """Parse content request for summarization"""
        try:
            return json.loads(content_input)
        except:
            return {"content": content_input}
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content"""
        blob = TextBlob(content)
        return [str(sentence) for sentence in blob.sentences]
    
    def _create_bullet_summary(self, sentences: List[str], max_length: int) -> str:
        """Create bullet point summary"""
        key_sentences = sentences[:5]  # Simple approach - take first 5 sentences
        return "\n".join([f"• {sentence.strip()}" for sentence in key_sentences])
    
    def _create_paragraph_summary(self, sentences: List[str], max_length: int) -> str:
        """Create paragraph summary"""
        key_sentences = sentences[:3]  # Take first 3 sentences
        return " ".join(key_sentences)
    
    def _create_abstract_summary(self, sentences: List[str], max_length: int) -> str:
        """Create abstract summary"""
        return f"Abstract: {' '.join(sentences[:2])}"
    
    def _create_key_points_summary(self, sentences: List[str], max_length: int) -> str:
        """Create key points summary"""
        return f"Key Points:\n" + "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(sentences[:3])])
    
    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        # Simple approach - extract important words
        blob = TextBlob(content)
        important_words = [word for word in blob.words if len(word) > 4]
        return list(set(important_words[:10]))
    
    def _parse_translation_request(self, request: str) -> Dict[str, Any]:
        """Parse translation request"""
        try:
            return json.loads(request)
        except:
            return {"text": request}
    
    def _parse_format_request(self, request: str) -> Dict[str, Any]:
        """Parse format request"""
        try:
            return json.loads(request)
        except:
            return {"content": request}
    
    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown"""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
            elif line.isupper() and len(line) < 50:
                markdown_lines.append(f"# {line.title()}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)
    
    def _parse_tone_request(self, request: str) -> Dict[str, Any]:
        """Parse tone analysis request"""
        try:
            return json.loads(request)
        except:
            return {"text": request}
    
    def _detect_tone(self, text: str) -> str:
        """Detect current tone of text"""
        # Simple tone detection based on patterns
        if any(word in text.lower() for word in ['please', 'thank', 'appreciate', 'kindly']):
            return "polite"
        elif any(word in text.lower() for word in ['urgent', 'immediately', 'asap', 'critical']):
            return "urgent"
        elif '!' in text:
            return "enthusiastic"
        else:
            return "neutral"
    
    def _suggest_tone_adjustments(self, text: str, current_tone: str, desired_tone: str) -> List[str]:
        """Suggest tone adjustments"""
        adjustments = []
        
        if desired_tone == "professional" and current_tone == "casual":
            adjustments.append("Use more formal language")
            adjustments.append("Avoid contractions")
        elif desired_tone == "friendly" and current_tone == "formal":
            adjustments.append("Use warmer language")
            adjustments.append("Add personal touches")
        
        return adjustments
    
    def _adjust_text_tone(self, text: str, desired_tone: str) -> str:
        """Adjust text tone"""
        # This is a simplified version - would need more sophisticated processing
        if desired_tone == "professional":
            return text.replace("can't", "cannot").replace("won't", "will not")
        elif desired_tone == "friendly":
            return f"Hi there! {text} Hope this helps!"
        return text
    
    def _analyze_formality(self, text: str) -> str:
        """Analyze formality level"""
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'sincerely']
        casual_indicators = ['hey', 'yeah', 'gonna', 'wanna']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        casual_count = sum(1 for word in casual_indicators if word in text.lower())
        
        if formal_count > casual_count:
            return "formal"
        elif casual_count > formal_count:
            return "casual"
        else:
            return "neutral"
    
    def _analyze_emotional_intensity(self, text: str) -> str:
        """Analyze emotional intensity"""
        if '!' in text or text.isupper():
            return "high"
        elif '?' in text:
            return "medium"
        else:
            return "low"
    
    def _analyze_confidence(self, text: str) -> str:
        """Analyze confidence level"""
        confident_words = ['will', 'definitely', 'certainly', 'absolutely']
        uncertain_words = ['might', 'maybe', 'perhaps', 'possibly']
        
        confident_count = sum(1 for word in confident_words if word in text.lower())
        uncertain_count = sum(1 for word in uncertain_words if word in text.lower())
        
        if confident_count > uncertain_count:
            return "high"
        elif uncertain_count > confident_count:
            return "low"
        else:
            return "medium"
    
    def _parse_presentation_request(self, request: str) -> Dict[str, Any]:
        """Parse presentation request"""
        try:
            return json.loads(request)
        except:
            return {"topic": request}
    
    def _create_presentation_outline(self, topic: str, slide_count: int) -> List[str]:
        """Create presentation outline"""
        return [
            "1. Introduction",
            "2. Problem Statement",
            "3. Methodology",
            "4. Key Findings",
            "5. Analysis",
            "6. Recommendations",
            "7. Next Steps",
            "8. Q&A"
        ][:slide_count]
    
    def _create_presentation_slides(self, topic: str, audience: str, slide_count: int, data: Dict) -> List[Dict]:
        """Create presentation slides"""
        slides = []
        outline = self._create_presentation_outline(topic, slide_count)
        
        for i, slide_title in enumerate(outline):
            slides.append({
                "slide_number": i + 1,
                "title": slide_title,
                "content": f"Content for {slide_title} related to {topic}",
                "notes": f"Speaker notes for slide {i + 1}"
            })
        
        return slides
    
    def _parse_social_request(self, request: str) -> Dict[str, Any]:
        """Parse social media request"""
        try:
            return json.loads(request)
        except:
            return {"topic": request}
    
    def _create_twitter_content(self, topic: str, content_type: str, tone: str) -> str:
        """Create Twitter content"""
        return f"🚀 Excited to share insights about {topic}! Key takeaways coming up in this thread 👇 #Innovation #Technology"
    
    def _create_linkedin_content(self, topic: str, content_type: str, tone: str) -> str:
        """Create LinkedIn content"""
        return f"I've been reflecting on {topic} and wanted to share some insights with my network.\n\nKey observations:\n• Innovation drives progress\n• Collaboration is essential\n• Continuous learning matters\n\nWhat are your thoughts? #Leadership #Innovation"
    
    def _create_facebook_content(self, topic: str, content_type: str, tone: str) -> str:
        """Create Facebook content"""
        return f"Sharing some thoughts on {topic}. This is such an important area that affects all of us. Would love to hear your perspectives in the comments!"
    
    def _create_instagram_content(self, topic: str, content_type: str, tone: str) -> str:
        """Create Instagram content"""
        return f"✨ {topic} ✨\n\nSometimes the best insights come from unexpected places. Here's what I learned today...\n\n#Inspiration #Learning #Growth"
    
    def _create_general_social_content(self, topic: str, content_type: str, tone: str) -> str:
        """Create general social media content"""
        return f"Sharing insights about {topic}. Great discussion starter for any platform!"
    
    def _generate_hashtags(self, topic: str, platform: str) -> List[str]:
        """Generate relevant hashtags"""
        base_tags = ["#innovation", "#technology", "#insights"]
        topic_tags = [f"#{topic.replace(' ', '').lower()}"]
        return base_tags + topic_tags
    
    def _suggest_posting_time(self, platform: str) -> str:
        """Suggest optimal posting time"""
        times = {
            "twitter": "9:00 AM - 10:00 AM EST",
            "linkedin": "8:00 AM - 9:00 AM EST",
            "facebook": "1:00 PM - 3:00 PM EST",
            "instagram": "11:00 AM - 1:00 PM EST"
        }
        return times.get(platform, "Business hours")
    
    def _get_engagement_tips(self, platform: str) -> List[str]:
        """Get platform-specific engagement tips"""
        tips = {
            "twitter": ["Use relevant hashtags", "Engage in conversations", "Share threads"],
            "linkedin": ["Tag relevant people", "Ask questions", "Share professional insights"],
            "facebook": ["Use engaging visuals", "Ask for comments", "Share personal stories"],
            "instagram": ["Use high-quality images", "Write engaging captions", "Use stories"]
        }
        return tips.get(platform, ["Create engaging content", "Be consistent", "Respond to comments"])