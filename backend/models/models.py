from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from core.database import Base
import uuid
from datetime import datetime


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    routes = relationship("Route", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")


class Route(Base):
    __tablename__ = "routes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Route metadata
    name = Column(String, nullable=True)
    start_latitude = Column(Float, nullable=False)
    start_longitude = Column(Float, nullable=False)
    target_distance = Column(Integer, nullable=False)  # in meters
    actual_distance = Column(Float, nullable=False)   # in meters
    
    # Route data
    path_nodes = Column(JSON, nullable=False)  # List of node IDs
    gpx_data = Column(Text, nullable=True)     # GPX export
    elevation_gain = Column(Float, nullable=True)
    
    # Semantic data
    preference_query = Column(Text, nullable=True)    # User's preference description
    matched_places = Column(JSON, nullable=True)      # Places that influenced the route
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="routes")


class Place(Base):
    __tablename__ = "places"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic place info
    name = Column(String, nullable=False)
    place_type = Column(String, nullable=False)  # park, lake, cafe, etc.
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # OSM data
    osm_id = Column(String, nullable=True, index=True)
    osm_type = Column(String, nullable=True)  # node, way, relation
    
    # Semantic data
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)           # OSM tags
    embedding = Column(LargeBinary, nullable=True)  # Stored as binary numpy array
    
    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Spatial index for location queries would be added here


class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Preference data
    description = Column(Text, nullable=False)  # "I like quiet parks near water"
    embedding = Column(LargeBinary, nullable=True)  # Preference embedding
    weight = Column(Float, default=1.0)        # How much to weight this preference
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="preferences")


class RouteCache(Base):
    __tablename__ = "route_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cache key components
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    distance = Column(Integer, nullable=False)
    preference_hash = Column(String, nullable=True)  # Hash of preference embedding
    
    # Cached route data
    cached_route = Column(JSON, nullable=False)
    
    # Cache metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    hit_count = Column(Integer, default=0)
    
    # TTL for cache invalidation
    expires_at = Column(DateTime, nullable=False)