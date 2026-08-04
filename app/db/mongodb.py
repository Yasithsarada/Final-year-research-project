import logging
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from app.db.base import BaseDatabase
from app.models.resume import ResumeDocument
from app.models.ccs import CCSJobDocument
from app.models.candidate_profile import CandidateProfileDocument
from app.core.config import settings
from app.core.exceptions import DatabaseException

logger = logging.getLogger(__name__)

class MongoDBDatabase(BaseDatabase):
    """MongoDB implementation of the Database interface using motor (async)."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection_name = "resumes"
        self.ccs_collection_name = "ccs_jobs"
        self.profile_collection_name = "candidate_profiles"

    async def connect(self) -> None:
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}...")
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.client[settings.DB_NAME]
            # Verify connection by triggering a simple admin command
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise DatabaseException(f"MongoDB connection error: {str(e)}")

    async def disconnect(self) -> None:
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection.")

    async def save_resume(self, document: ResumeDocument) -> str:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            # Prepare document dict
            doc_dict = document.model_dump(by_alias=True)
            
            # Remove id if it is None or empty so MongoDB autogenerates one
            if "_id" in doc_dict and doc_dict["_id"] is None:
                doc_dict.pop("_id")
                
            # Perform upsert or insert
            if document.id:
                # Update existing
                result = await self.db[self.collection_name].update_one(
                    {"_id": document.id},
                    {"$set": doc_dict},
                    upsert=True
                )
                inserted_id = document.id
            else:
                # Insert new
                result = await self.db[self.collection_name].insert_one(doc_dict)
                inserted_id = str(result.inserted_id)
                
            return inserted_id
        except Exception as e:
            logger.error(f"Failed to save resume: {str(e)}")
            raise DatabaseException(f"Failed to write resume to database: {str(e)}")

    async def get_resume(self, resume_id: str) -> Optional[ResumeDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            # Resume ID might be standard string (UUID) or MongoDB ObjectId
            query = {"_id": resume_id}
            doc = await self.db[self.collection_name].find_one(query)
            
            if not doc:
                # Try finding as ObjectId
                try:
                    query = {"_id": ObjectId(resume_id)}
                    doc = await self.db[self.collection_name].find_one(query)
                except Exception:
                    pass
                    
            if not doc:
                return None
                
            # Map _id object to string for Pydantic parsing
            if isinstance(doc.get("_id"), ObjectId):
                doc["_id"] = str(doc["_id"])
                
            return ResumeDocument.model_validate(doc)
        except Exception as e:
            logger.error(f"Failed to fetch resume {resume_id}: {str(e)}")
            raise DatabaseException(f"Failed to read resume from database: {str(e)}")

    async def list_resumes(self, limit: int = 20, skip: int = 0) -> List[ResumeDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            cursor = self.db[self.collection_name].find().skip(skip).limit(limit)
            resumes = []
            
            async for doc in cursor:
                if isinstance(doc.get("_id"), ObjectId):
                    doc["_id"] = str(doc["_id"])
                resumes.append(ResumeDocument.model_validate(doc))
                
            return resumes
        except Exception as e:
            logger.error(f"Failed to list resumes: {str(e)}")
            raise DatabaseException(f"Failed to query resumes database: {str(e)}")

    async def save_ccs_job(self, document: CCSJobDocument) -> str:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            doc_dict = document.model_dump(by_alias=True)
            if "_id" in doc_dict and doc_dict["_id"] is None:
                doc_dict.pop("_id")
                
            if document.id:
                result = await self.db[self.ccs_collection_name].update_one(
                    {"_id": document.id},
                    {"$set": doc_dict},
                    upsert=True
                )
                inserted_id = document.id
            else:
                result = await self.db[self.ccs_collection_name].insert_one(doc_dict)
                inserted_id = str(result.inserted_id)
                
            return inserted_id
        except Exception as e:
            logger.error(f"Failed to save CCS job: {str(e)}")
            raise DatabaseException(f"Failed to write CCS job to database: {str(e)}")

    async def get_ccs_job(self, job_id: str) -> Optional[CCSJobDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            query = {"_id": job_id}
            doc = await self.db[self.ccs_collection_name].find_one(query)
            
            if not doc:
                try:
                    query = {"_id": ObjectId(job_id)}
                    doc = await self.db[self.ccs_collection_name].find_one(query)
                except Exception:
                    pass
                    
            if not doc:
                return None
                
            if isinstance(doc.get("_id"), ObjectId):
                doc["_id"] = str(doc["_id"])
                
            return CCSJobDocument.model_validate(doc)
        except Exception as e:
            logger.error(f"Failed to fetch CCS job {job_id}: {str(e)}")
            raise DatabaseException(f"Failed to read CCS job from database: {str(e)}")

    async def list_ccs_jobs(self, limit: int = 20, skip: int = 0) -> List[CCSJobDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
            
        try:
            cursor = self.db[self.ccs_collection_name].find().skip(skip).limit(limit)
            jobs = []
            
            async for doc in cursor:
                if isinstance(doc.get("_id"), ObjectId):
                    doc["_id"] = str(doc["_id"])
                jobs.append(CCSJobDocument.model_validate(doc))
                
            return jobs
        except Exception as e:
            logger.error(f"Failed to list CCS jobs: {str(e)}")
            raise DatabaseException(f"Failed to query CCS jobs database: {str(e)}")

    async def save_candidate_profile(self, document: CandidateProfileDocument) -> str:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
        try:
            doc_dict = document.model_dump(by_alias=True)
            if "_id" in doc_dict and doc_dict["_id"] is None:
                doc_dict.pop("_id")
            if document.id:
                await self.db[self.profile_collection_name].update_one(
                    {"_id": document.id}, {"$set": doc_dict}, upsert=True
                )
                return document.id
            else:
                result = await self.db[self.profile_collection_name].insert_one(doc_dict)
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save candidate profile: {str(e)}")
            raise DatabaseException(f"Failed to write candidate profile to database: {str(e)}")

    async def get_candidate_profile(self, profile_id: str) -> Optional[CandidateProfileDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
        try:
            doc = await self.db[self.profile_collection_name].find_one({"_id": profile_id})
            if not doc:
                try:
                    doc = await self.db[self.profile_collection_name].find_one({"_id": ObjectId(profile_id)})
                except Exception:
                    pass
            if not doc:
                return None
            if isinstance(doc.get("_id"), ObjectId):
                doc["_id"] = str(doc["_id"])
            return CandidateProfileDocument.model_validate(doc)
        except Exception as e:
            logger.error(f"Failed to fetch candidate profile {profile_id}: {str(e)}")
            raise DatabaseException(f"Failed to read candidate profile from database: {str(e)}")

    async def list_candidate_profiles(self, limit: int = 20, skip: int = 0) -> List[CandidateProfileDocument]:
        if not self.db:
            raise DatabaseException("Database client is not connected.")
        try:
            cursor = self.db[self.profile_collection_name].find().skip(skip).limit(limit)
            profiles = []
            async for doc in cursor:
                if isinstance(doc.get("_id"), ObjectId):
                    doc["_id"] = str(doc["_id"])
                profiles.append(CandidateProfileDocument.model_validate(doc))
            return profiles
        except Exception as e:
            logger.error(f"Failed to list candidate profiles: {str(e)}")
            raise DatabaseException(f"Failed to query candidate profiles database: {str(e)}")
