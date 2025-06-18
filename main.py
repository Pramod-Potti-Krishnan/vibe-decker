"""
Vibe Decker Webhook API - Streamlined Core Version
Optimized for frontend consumption with essential features only
"""

import os
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
from dotenv import load_dotenv # <--- ADD THIS LINE
load_dotenv() # <--- AND ADD THIS LINE

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Google services
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# OpenAI
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELS - Request/Response schemas for the API
# ============================================================================

class PresentationRequest(BaseModel):
    """Request model for presentation generation"""
    content: str = Field(..., description="Content to convert to slides", min_length=50)
    audience: str = Field(default="stakeholders", description="Target audience")
    tone: str = Field(default="professional", description="Presentation tone")
    user_email: Optional[str] = Field(None, description="User email for Google auth")

class PresentationResponse(BaseModel):
    """Response model for presentation requests"""
    session_id: str
    status: str  # "processing" | "completed" | "failed" | "pending_auth"
    message: str
    slides_url: Optional[str] = None
    preview_data: Optional[List[Dict]] = None
    auth_url: Optional[str] = None

class SessionStatus(BaseModel):
    """Status check response"""
    session_id: str
    status: str
    progress: int  # 0-100
    message: str
    slides_url: Optional[str] = None
    error: Optional[str] = None

class SlideData(BaseModel):
    """Individual slide data"""
    slide_number: int
    title: str
    content: List[str]
    slide_type: str

# ============================================================================
# SESSION MANAGEMENT - Simple in-memory store
# ============================================================================

@dataclass
class Session:
    id: str
    status: str
    progress: int
    message: str
    request_data: Dict
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# In-memory session store (use Redis in production)
sessions: Dict[str, Session] = {}

# ============================================================================
# CORE VIBE DECKER AGENT - Streamlined Version
# ============================================================================

class CoreVibeDeckerAgent:
    """Streamlined core agent for slide generation"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.service = None
        
    def set_google_service(self, credentials):
        """Set Google Slides service with provided credentials"""
        self.service = build('slides', 'v1', credentials=credentials)
    
    async def analyze_content(self, content: str, audience: str = "stakeholders", tone: str = "professional") -> List[Dict]:
        """Analyze content and generate slide structure"""
        try:
            prompt = f"""
            Analyze this content and create a presentation structure for {audience} with a {tone} tone.
            
            Content: {content}
            
            Create 5-8 slides with:
            1. Title slide
            2. Content slides with clear titles and 3-5 bullet points each
            3. Conclusion slide
            
            Return as JSON array with this structure:
            [
                {{
                    "slide_number": 1,
                    "title": "Slide Title",
                    "content": ["bullet point 1", "bullet point 2", ...],
                    "slide_type": "title|content|conclusion"
                }}
            ]
            
            Keep titles concise and bullet points actionable. Focus on key insights.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse JSON response
            import json
            slides_data = json.loads(response.choices[0].message.content)
            
            logger.info(f"Generated {len(slides_data)} slides")
            return slides_data
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")
    
    async def create_google_slides(self, title: str = "AI-Generated Presentation") -> str:
        """Create a new Google Slides presentation"""
        try:
            presentation = {
                'title': title
            }
            
            response = self.service.presentations().create(body=presentation).execute()
            presentation_id = response['presentationId']
            
            logger.info(f"Created presentation: {presentation_id}")
            return presentation_id
            
        except Exception as e:
            logger.error(f"Failed to create slides: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create slides: {str(e)}")
    
    async def populate_slides(self, presentation_id: str, slides_data: List[Dict]) -> str:
        """Populate slides with content"""
        try:
            # Get existing presentation
            presentation = self.service.presentations().get(presentationId=presentation_id).execute()
            existing_slides = presentation.get('slides', [])
            
            # Update title slide if it exists
            if existing_slides and slides_data:
                await self._update_title_slide(presentation_id, existing_slides[0], slides_data[0])
                slides_to_create = slides_data[1:]  # Skip title slide
            else:
                slides_to_create = slides_data
            
            # Create content slides
            for slide_data in slides_to_create:
                await self._create_content_slide(presentation_id, slide_data)
            
            # Generate shareable URL
            slides_url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"
            
            logger.info(f"Populated {len(slides_data)} slides")
            return slides_url
            
        except Exception as e:
            logger.error(f"Failed to populate slides: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to populate slides: {str(e)}")
    
    async def _update_title_slide(self, presentation_id: str, title_slide: Dict, title_data: Dict):
        """Update the existing title slide"""
        try:
            requests = []
            page_elements = title_slide.get('pageElements', [])
            
            # Find title element (try multiple strategies)
            title_element = None
            for element in page_elements:
                if element.get('shape') and element['shape'].get('placeholder'):
                    placeholder_type = element['shape']['placeholder'].get('type')
                    if placeholder_type in ['TITLE', 'CENTERED_TITLE']:
                        title_element = element['objectId']
                        break
            
            # Fallback: use first text box
            if not title_element:
                for element in page_elements:
                    if element.get('shape') and element['shape'].get('shapeType') == 'TEXT_BOX':
                        title_element = element['objectId']
                        break
            
            if title_element:
                requests.append({
                    'insertText': {
                        'objectId': title_element,
                        'text': title_data['title'],
                        'insertionIndex': 0
                    }
                })
                
                # Execute update
                self.service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': requests}
                ).execute()
                
        except Exception as e:
            logger.warning(f"Title slide update failed: {str(e)}")
    
    async def _create_content_slide(self, presentation_id: str, slide_data: Dict):
        """Create a content slide with title and bullets"""
        try:
            # Create slide
            slide_id = f"slide_{uuid.uuid4().hex[:8]}"
            requests = [{
                'createSlide': {
                    'objectId': slide_id,
                    'slideLayoutReference': {'predefinedLayout': 'TITLE_AND_BODY'}
                }
            }]
            
            # Execute slide creation
            self.service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': requests}
            ).execute()
            
            # Add content
            content_requests = []
            
            # Add title
            title_id = f"title_{uuid.uuid4().hex[:8]}"
            content_requests.extend([
                {
                    'createShape': {
                        'objectId': title_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': slide_id,
                            'size': {'height': {'magnitude': 50, 'unit': 'PT'}, 'width': {'magnitude': 600, 'unit': 'PT'}},
                            'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 50, 'translateY': 50, 'unit': 'PT'}
                        }
                    }
                },
                {
                    'insertText': {
                        'objectId': title_id,
                        'text': slide_data['title'],
                        'insertionIndex': 0
                    }
                }
            ])
            
            # Add bullet points
            if slide_data.get('content'):
                body_id = f"body_{uuid.uuid4().hex[:8]}"
                bullet_text = '\n'.join([f"• {point}" for point in slide_data['content']])
                
                content_requests.extend([
                    {
                        'createShape': {
                            'objectId': body_id,
                            'shapeType': 'TEXT_BOX',
                            'elementProperties': {
                                'pageObjectId': slide_id,
                                'size': {'height': {'magnitude': 300, 'unit': 'PT'}, 'width': {'magnitude': 600, 'unit': 'PT'}},
                                'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 50, 'translateY': 120, 'unit': 'PT'}
                            }
                        }
                    },
                    {
                        'insertText': {
                            'objectId': body_id,
                            'text': bullet_text,
                            'insertionIndex': 0
                        }
                    }
                ])
            
            # Execute content addition
            self.service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': content_requests}
            ).execute()
            
        except Exception as e:
            logger.error(f"Failed to create content slide: {str(e)}")

# ============================================================================
# GOOGLE AUTHENTICATION HELPERS
# ============================================================================

def get_google_auth_flow():
    """Create Google OAuth flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8000/auth/callback')]
            }
        },
        scopes=['https://www.googleapis.com/auth/presentations']
    )
    flow.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8000/auth/callback')
    return flow

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Vibe Decker API",
    description="Streamlined presentation generation webhook",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
# agent = CoreVibeDeckerAgent()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/generate-presentation", response_model=PresentationResponse)
async def generate_presentation(
    request: PresentationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a presentation from content"""
    try:
        # Create session
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            status="pending_auth",
            progress=0,
            message="Waiting for Google authentication",
            request_data=request.dict()
        )
        sessions[session_id] = session
        
        # Generate auth URL
        ###flow = get_google_auth_flow()
        ###flow.redirect_uri = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/auth/callback/{session_id}"
        ###auth_url, _ = flow.authorization_url(prompt='consent')
        # Generate auth URL
        flow = get_google_auth_flow()
        # The session_id is passed via the 'state' parameter, not the redirect_uri
        auth_url, _ = flow.authorization_url(prompt='consent', state=session_id)



        return PresentationResponse(
            session_id=session_id,
            status="pending_auth",
            message="Please complete Google authentication",
            auth_url=auth_url
        )
        
    except Exception as e:
        logger.error(f"Failed to start presentation generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

##@app.get("/auth/callback/{session_id}")
##async def auth_callback(session_id: str, code: str, background_tasks: BackgroundTasks):
@app.get("/auth/callback")
async def auth_callback(code: str, state: str, background_tasks: BackgroundTasks):
    session_id = state # The session_id is now the state

    """Handle Google OAuth callback"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # Exchange code for credentials
        flow = get_google_auth_flow()
        ##flow.redirect_uri = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/auth/callback/{session_id}"
        flow.fetch_token(code=code)
        
        credentials = flow.credentials
        
        # Start background processing
        background_tasks.add_task(process_presentation, session_id, credentials)
        
        # Update session
        session.status = "processing"
        session.progress = 10
        session.message = "Authentication complete. Generating slides..."
        
        return JSONResponse({
            "message": "Authentication successful. Processing presentation...",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Auth callback failed: {str(e)}")
        if session_id in sessions:
            sessions[session_id].status = "failed"
            sessions[session_id].error = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/presentation/{session_id}/status", response_model=SessionStatus)
async def get_presentation_status(session_id: str):
    """Get presentation generation status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionStatus(
        session_id=session_id,
        status=session.status,
        progress=session.progress,
        message=session.message,
        slides_url=session.result.get('slides_url') if session.result else None,
        error=session.error
    )

@app.get("/api/presentation/{session_id}/result")
async def get_presentation_result(session_id: str):
    """Get completed presentation result"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Presentation not ready")
    
    return session.result

# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

async def process_presentation(session_id: str, credentials):
    """Background task to process presentation"""
    try:
        agent = CoreVibeDeckerAgent() 
        session = sessions[session_id]
        request_data = session.request_data
        
        # Set up agent with credentials
        agent.set_google_service(credentials)
        
        # Step 1: Analyze content
        session.progress = 25
        session.message = "Analyzing content..."
        
        slides_data = await agent.analyze_content(
            request_data['content'],
            request_data.get('audience', 'stakeholders'),
            request_data.get('tone', 'professional')
        )
        
        # Step 2: Create slides
        session.progress = 50
        session.message = "Creating Google Slides..."
        
        presentation_id = await agent.create_google_slides(
            title=slides_data[0]['title'] if slides_data else "AI-Generated Presentation"
        )
        
        # Step 3: Populate content
        session.progress = 75
        session.message = "Adding content to slides..."
        
        slides_url = await agent.populate_slides(presentation_id, slides_data)
        
        # Complete
        session.status = "completed"
        session.progress = 100
        session.message = "Presentation ready!"
        session.result = {
            "slides_url": slides_url,
            "presentation_id": presentation_id,
            "slide_count": len(slides_data),
            "preview_data": slides_data
        }
        
    except Exception as e:
        logger.error(f"Presentation processing failed: {str(e)}")
        session.status = "failed"
        session.error = str(e)
        session.message = f"Failed: {str(e)}"

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Vibe Decker API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate": "/api/generate-presentation",
            "status": "/api/presentation/{session_id}/status",
            "result": "/api/presentation/{session_id}/result"
        }
    }

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
