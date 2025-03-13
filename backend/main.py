from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    func,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.ext.declarative import declared_attr
import os
import logging
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recipe Cost Prediction API",
    description="An API for prediciting recipe cost",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Base model with comon fields
class BaseDBModel:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Define RecipeCost Model
class RecipeCost(Base):
    __tablename__ = "recipe_cost"
    id = Column(Integer, primary_key=True, index=True)
    ingredient_cost = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    waste_percentage = Column(Float, nullable=False)
    processing_cost = Column(Float, nullable=False)
    cost_per_unit = Column(Float, nullable=False)
    labor_hours = Column(Float, nullable=True)
    labor_rate = Column(Float, nullable=True)
    overhead_cost = Column(Float, nullable=True)
    packaging_cost = Column(Float, nullable=True)
    recipe_name = Column(String(255), nullable=True)
    recipe_category = Column(String(100), nullable=True)
    cost_per_unit = Column(Float, nullable=False)
    profit_margin = Column(Float, nullable=True)
    suggested_price = Column(Float, nullable=True)


# Prediction History
class PredictionHistory(Base, BaseDBModel):
    __tablename__ = "prediction_history"
    input_data = Column(String, nullable=False)
    predicted_cost = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    accuracy_metric = Column(Float, nullable=True)


# Model Perfomance
class ModelPerformance(Base, BaseDBModel):
    __tablename__ = "model_performance"
    model_type = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    training_samples = Column(Integer, nullable=False)
    mae = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    cross_val_score = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)


class PriceCalculationInput(BaseModel):
    recipe_id: int
    target_margin: float = Field(
        ..., gt=0, le=100, description="Target profit margin percentage"
    )


# Create tables
Base.metadata.create_all(bind=engine)


# Request Model
class RecipeInput(BaseModel):
    ingredient_cost: float = Field(..., gt=0, description="Total cost of ingredients")
    quantity: float = Field(..., gt=0, description="Number of units produced")
    waste_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of waste during production"
    )
    processing_cost: float = Field(
        ..., ge=0, description="Cost of processing ingredients"
    )

    # Optional fields
    labor_hours: Optional[float] = Field(
        0.0, ge=0, description="Hours of labor required"
    )
    labor_rate: Optional[float] = Field(0.0, ge=0, description="Hourly labor rate")
    overhead_cost: Optional[float] = Field(
        0.0, ge=0, description="Additional overhead costs"
    )
    packaging_cost: Optional[float] = Field(
        0.0, ge=0, description="Cost of packaging per unit"
    )
    recipe_name: Optional[str] = Field(None, description="Name of the recipe")
    recipe_category: Optional[str] = Field(None, description="Category of the recipe")

    @validator("waste_percentage")
    def validate_waste_percentage(cls, value):
        if value < 0 or value > 100:
            raise ValueError("Waste percentage must be between 0 and 100")
        return value


class ProfitMarginInput(BaseModel):
    target_margin: float = Field(
        ..., gt=0, le=100, description="Target profit margin percentage"
    )


class RecipeSearch(BaseModel):
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    category: Optional[str] = None
    name_contains: Optional[str] = None


# Model handling functions
class ModelManager:
    MODEL_DIR = "models"
    CURRENT_MODEL_INFO = "current_model.json"

    def __init__(self):
        # Create models directory if not exists
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.current_model_info = self._load_model_info()

    def _load_model_info(self):
        info_path = os.path.join(self.MODEL_DIR, self.CURRENT_MODEL_INFO)
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    return json.load(f)
            except:
                pass

        # Default info if none exists
        return {
            "current_model": "random_forest",
            "version": "1.0.0",
            "last_trained": datetime.now().isoformat(),
            "features": [
                "ingredient_cost",
                "quantity",
                "waste_percentage",
                "processing_cost",
            ],
        }

    def _save_model_info(self, info):
        with open(os.path.join(self.MODEL_DIR, self.CURRENT_MODEL_INFO), "w") as f:
            json.dump(info, f)
        self.current_model_info = info

    def get_model_path(self, model_type="random_forest"):
        return os.path.join(self.MODEL_DIR, f"{model_type}_model.pkl")

    def get_current_model_type(self):
        return self.current_model_info.get("current_model", "random_forest")

    def get_model_features(self):
        return self.current_model_info.get(
            "features",
            ["ingredient_cost", "quantity", "waste_percentage", "processing_cost"],
        )

    def save_model(
        self, model, model_type="random_forest", metrics=None, features=None
    ):
        model_path = self.get_model_path(model_type)

        # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Update model info
        self.current_model_info["current_model"] = model_type
        self.current_model_info["version"] = str(uuid.uuid4())[:8]
        self.current_model_info["last_trained"] = datetime.now().isoformat()

        if features:
            self.current_model_info["features"] = features

        if metrics:
            self.current_model_info["metrics"] = metrics

        self._save_model_info(self.current_model_info)

        return self.current_model_info

    def load_model(self, model_type=None):
        if model_type is None:
            model_type = self.get_current_model_type()

        model_path = self.get_model_path(model_type)

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)

        return None


# Initialize the model manager
model_manager = ModelManager()


def train_models(db: Session):
    """Train multiple models and select the best one"""
    start_time = time.time()

    # Get training data
    training_data = get_training_data()
    if training_data.empty:
        logger.warning("No training data available. Creating fallback model.")
        # Create and save a fallback model
        return _create_fallback_model(db)

    # Determine which features to use (including new ones if available)
    all_columns = training_data.columns.tolist()
    feature_columns = [col for col in all_columns if col != "cost_per_unit"]

    X = training_data[feature_columns]
    y = training_data["cost_per_unit"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models to test
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "linear_regression": LinearRegression(),
    }

    best_model = None
    best_model_type = None
    best_score = float("-inf")
    model_metrics = {}

    # Train and evaluate each model
    for model_type, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate cross-validation score
        cv_score = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        # Record metrics
        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2),
            "cv_score": float(cv_score),
        }
        model_metrics[model_type] = metrics

        # Save model performance to database
        performance = ModelPerformance(
            model_type=model_type,
            model_version=str(uuid.uuid4())[:8],
            training_samples=len(X_train),
            mae=mae,
            mse=mse,
            r2=r2,
            cross_val_score=cv_score,
            training_time=time.time() - start_time,
        )
        db.add(performance)

        # Check if this is the best model
        if cv_score > best_score:
            best_score = cv_score
            best_model = model
            best_model_type = model_type

    # Commit performance records
    db.commit()

    # Save the best model
    if best_model:
        logger.info(f"Best model: {best_model_type} with cv_score: {best_score}")
        model_info = model_manager.save_model(
            best_model,
            model_type=best_model_type,
            metrics=model_metrics[best_model_type],
            features=feature_columns,
        )
        return best_model, model_info

    # If we somehow didn't get a best model, create a fallback
    return _create_fallback_model(db)


def _create_fallback_model(db: Session):
    """Create a simple fallback model when no training data is available"""
    # Create a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Default features
    features = ["ingredient_cost", "quantity", "waste_percentage", "processing_cost"]

    # Train with default data
    default_X = np.array(
        [
            [10.0, 2.0, 5.0, 3.0],
            [15.0, 3.0, 7.0, 4.0],
            [20.0, 4.0, 10.0, 5.0],
            [25.0, 5.0, 12.0, 6.0],
        ]
    )
    default_y = np.array([10.5, 17.8, 25.2, 33.0])

    model.fit(default_X, default_y)

    # Create metrics
    metrics = {
        "mae": 0.0,  # Default values
        "mse": 0.0,
        "r2": 1.0,
        "cv_score": 1.0,
        "is_fallback": True,
    }

    # Save performance to database
    performance = ModelPerformance(
        model_type="random_forest_fallback",
        model_version="fallback_1.0",
        training_samples=4,
        mae=0.0,
        mse=0.0,
        r2=1.0,
        cross_val_score=1.0,
    )
    db.add(performance)
    db.commit()

    # Save the model
    model_info = model_manager.save_model(
        model, model_type="random_forest", metrics=metrics, features=features
    )

    return model, model_info


def load_or_train_model(db: Session = None):
    """Load the current model or train a new one if needed"""
    # Try to load existing model
    model = model_manager.load_model()

    if model is not None:
        return model, model_manager.current_model_info

    # If no model exists, train a new one
    if db is not None:
        return train_models(db)

    # If no DB session is provided, create one
    db = SessionLocal()
    try:
        return train_models(db)
    finally:
        db.close()


# Load model at startup
@app.on_event("startup")
async def startup_event():
    try:
        db = SessionLocal()
        try:
            load_or_train_model(db)
            logger.info("Model loaded successfully on startup")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.post("/seed", status_code=status.HTTP_201_CREATED)
def seed_initial_data(db: Session = Depends(get_db)):
    try:
        # Check if there's already data
        existing_data = db.query(RecipeCost).count()
        if existing_data > 0:
            return {
                "message": f"Database already contains {existing_data} records. No seeding needed."
            }

        # Enhanced sample data for initial training
        sample_data = [
            {
                "ingredient_cost": 10.0,
                "quantity": 2.0,
                "waste_percentage": 5.0,
                "processing_cost": 3.0,
                "labor_hours": 0.5,
                "labor_rate": 15.0,
                "overhead_cost": 2.0,
                "packaging_cost": 1.0,
                "recipe_name": "Basic Bread",
                "recipe_category": "Bakery",
                "cost_per_unit": 10.5,
                "profit_margin": 30.0,
                "suggested_price": 13.65,
            },
            {
                "ingredient_cost": 15.0,
                "quantity": 3.0,
                "waste_percentage": 7.0,
                "processing_cost": 4.0,
                "labor_hours": 0.8,
                "labor_rate": 15.0,
                "overhead_cost": 2.5,
                "packaging_cost": 1.2,
                "recipe_name": "Cinnamon Roll",
                "recipe_category": "Bakery",
                "cost_per_unit": 17.8,
                "profit_margin": 35.0,
                "suggested_price": 24.03,
            },
            {
                "ingredient_cost": 20.0,
                "quantity": 4.0,
                "waste_percentage": 10.0,
                "processing_cost": 5.0,
                "labor_hours": 1.0,
                "labor_rate": 15.0,
                "overhead_cost": 3.0,
                "packaging_cost": 1.5,
                "recipe_name": "Veggie Soup",
                "recipe_category": "Soup",
                "cost_per_unit": 25.2,
                "profit_margin": 40.0,
                "suggested_price": 35.28,
            },
            {
                "ingredient_cost": 25.0,
                "quantity": 5.0,
                "waste_percentage": 12.0,
                "processing_cost": 6.0,
                "labor_hours": 1.2,
                "labor_rate": 15.0,
                "overhead_cost": 3.5,
                "packaging_cost": 2.0,
                "recipe_name": "Chocolate Cake",
                "recipe_category": "Dessert",
                "cost_per_unit": 33.0,
                "profit_margin": 45.0,
                "suggested_price": 47.85,
            },
            {
                "ingredient_cost": 12.0,
                "quantity": 2.5,
                "waste_percentage": 6.0,
                "processing_cost": 3.5,
                "labor_hours": 0.6,
                "labor_rate": 15.0,
                "overhead_cost": 2.2,
                "packaging_cost": 1.1,
                "recipe_name": "Cheese Croissant",
                "recipe_category": "Bakery",
                "cost_per_unit": 13.7,
                "profit_margin": 30.0,
                "suggested_price": 17.81,
            },
            {
                "ingredient_cost": 18.0,
                "quantity": 3.5,
                "waste_percentage": 8.0,
                "processing_cost": 4.5,
                "labor_hours": 0.9,
                "labor_rate": 15.0,
                "overhead_cost": 2.7,
                "packaging_cost": 1.3,
                "recipe_name": "Chicken Sandwich",
                "recipe_category": "Sandwich",
                "cost_per_unit": 21.3,
                "profit_margin": 35.0,
                "suggested_price": 28.76,
            },
            {
                "ingredient_cost": 22.0,
                "quantity": 4.5,
                "waste_percentage": 11.0,
                "processing_cost": 5.5,
                "labor_hours": 1.1,
                "labor_rate": 15.0,
                "overhead_cost": 3.2,
                "packaging_cost": 1.7,
                "recipe_name": "Beef Stew",
                "recipe_category": "Main",
                "cost_per_unit": 28.9,
                "profit_margin": 40.0,
                "suggested_price": 40.46,
            },
            {
                "ingredient_cost": 28.0,
                "quantity": 5.5,
                "waste_percentage": 13.0,
                "processing_cost": 6.5,
                "labor_hours": 1.3,
                "labor_rate": 15.0,
                "overhead_cost": 3.8,
                "packaging_cost": 2.2,
                "recipe_name": "Seafood Pasta",
                "recipe_category": "Main",
                "cost_per_unit": 37.2,
                "profit_margin": 45.0,
                "suggested_price": 53.94,
            },
        ]

        # Add sample data to database
        for item in sample_data:
            recipe = RecipeCost(**item)
            db.add(recipe)

        db.commit()

        # Retrain the model with the new data
        train_models(db)

        return {
            "message": f"Added {len(sample_data)} sample records to the database and retrained the model.",
            "categories": list(set(item["recipe_category"] for item in sample_data)),
        }
    except Exception as e:
        logger.error(f"Error seeding data: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=Dict[str, Any])
def predict_cost(
    recipe: RecipeInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        # Load model
        model, model_info = load_or_train_model(db)

        # Get required features based on model info
        required_features = model_manager.get_model_features()

        # Create input data dictionary with all available fields
        input_dict = recipe.dict()

        # Filter and prepare input data based on required features
        input_data = np.array(
            [[input_dict.get(feature, 0) for feature in required_features]]
        )

        # Make prediction
        predicted_cost = float(model.predict(input_data)[0])

        # Calculate suggested price based on default margin if not specified
        profit_margin = 30.0  # Default 30% margin
        suggested_price = predicted_cost * (1 + profit_margin / 100)

        # Save prediction to DB
        new_recipe = RecipeCost(
            ingredient_cost=recipe.ingredient_cost,
            quantity=recipe.quantity,
            waste_percentage=recipe.waste_percentage,
            processing_cost=recipe.processing_cost,
            labor_hours=recipe.labor_hours,
            labor_rate=recipe.labor_rate,
            overhead_cost=recipe.overhead_cost,
            packaging_cost=recipe.packaging_cost,
            recipe_name=recipe.recipe_name,
            recipe_category=recipe.recipe_category,
            cost_per_unit=predicted_cost,
            profit_margin=profit_margin,
            suggested_price=suggested_price,
        )

        db.add(new_recipe)
        db.commit()
        db.refresh(new_recipe)

        # Log prediction in history (background task to not slow down response)
        background_tasks.add_task(
            log_prediction,
            db=db,
            input_data=json.dumps(input_dict),
            predicted_cost=predicted_cost,
            model_version=model_info.get("version", "unknown"),
        )

        return {
            "predicted_cost_per_unit": predicted_cost,
            "suggested_price": suggested_price,
            "profit_margin": profit_margin,
            "model_version": model_info.get("version", "unknown"),
            "model_type": model_info.get("current_model", "unknown"),
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def log_prediction(
    db: Session, input_data: str, predicted_cost: float, model_version: str
):
    """Log prediction to history table (called as background task)"""
    try:
        history = PredictionHistory(
            input_data=input_data,
            predicted_cost=predicted_cost,
            model_version=model_version,
        )
        db.add(history)
        db.commit()
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")
        db.rollback()


@app.post("/calculate-price", response_model=Dict[str, Any])
def calculate_price(
    calculation_input: PriceCalculationInput, db: Session = Depends(get_db)
):
    """Calculate suggested price based on target profit margin"""
    try:
        # Get recipe from database
        recipe = (
            db.query(RecipeCost)
            .filter(RecipeCost.id == calculation_input.recipe_id)
            .first()
        )
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Calculate suggested price based on target margin
        target_margin = calculation_input.target_margin
        suggested_price = recipe.cost_per_unit * (1 + target_margin / 100)

        # Update recipe in database
        recipe.profit_margin = target_margin
        recipe.suggested_price = suggested_price
        db.commit()

        return {
            "recipe_id": recipe.id,
            "recipe_name": recipe.recipe_name,
            "cost_per_unit": recipe.cost_per_unit,
            "profit_margin": target_margin,
            "suggested_price": suggested_price,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recipes", response_model=List[Dict[str, Any]])
def list_recipes(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    db: Session = Depends(get_db),
):
    """List recipes with optional filtering"""
    try:
        query = db.query(RecipeCost)

        # Apply filters
        if category:
            query = query.filter(RecipeCost.recipe_category == category)
        if min_cost is not None:
            query = query.filter(RecipeCost.cost_per_unit >= min_cost)
        if max_cost is not None:
            query = query.filter(RecipeCost.cost_per_unit <= max_cost)

        # Execute query with pagination
        recipes = query.offset(skip).limit(limit).all()

        # Convert to list of dictionaries
        result = []
        for recipe in recipes:
            recipe_dict = {
                "id": recipe.id,
                "recipe_name": recipe.recipe_name,
                "recipe_category": recipe.recipe_category,
                "cost_per_unit": recipe.cost_per_unit,
                "profit_margin": recipe.profit_margin,
                "suggested_price": recipe.suggested_price,
                "ingredient_cost": recipe.ingredient_cost,
                "quantity": recipe.quantity,
                "waste_percentage": recipe.waste_percentage,
                "processing_cost": recipe.processing_cost,
            }
            result.append(recipe_dict)

        return result
    except Exception as e:
        logger.error(f"Error listing recipes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recipe/{recipe_id}", response_model=Dict[str, Any])
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    """Get details of a specific recipe"""
    try:
        recipe = db.query(RecipeCost).filter(RecipeCost.id == recipe_id).first()
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        return {
            "id": recipe.id,
            "recipe_name": recipe.recipe_name,
            "recipe_category": recipe.recipe_category,
            "cost_per_unit": recipe.cost_per_unit,
            "profit_margin": recipe.profit_margin,
            "suggested_price": recipe.suggested_price,
            "ingredient_cost": recipe.ingredient_cost,
            "quantity": recipe.quantity,
            "waste_percentage": recipe.waste_percentage,
            "processing_cost": recipe.processing_cost,
            "labor_hours": recipe.labor_hours,
            "labor_rate": recipe.labor_rate,
            "overhead_cost": recipe.overhead_cost,
            "packaging_cost": recipe.packaging_cost,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/recipe/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    """Delete a recipe"""
    try:
        recipe = db.query(RecipeCost).filter(RecipeCost.id == recipe_id).first()
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        db.delete(recipe)
        db.commit()

        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recipe: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/recipe/{recipe_id}", response_model=Dict[str, Any])
def update_recipe(
    recipe_id: int, recipe_data: RecipeInput, db: Session = Depends(get_db)
):
    """Update a recipe"""
    try:
        recipe = db.query(RecipeCost).filter(RecipeCost.id == recipe_id).first()
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Update fields
        recipe_dict = recipe_data.dict(exclude_unset=True)
        for key, value in recipe_dict.items():
            setattr(recipe, key, value)

        # Re-predict cost with updated data
        model, model_info = load_or_train_model(db)
        required_features = model_manager.get_model_features()
        input_data = np.array(
            [[getattr(recipe, feature, 0) for feature in required_features]]
        )
        predicted_cost = float(model.predict(input_data)[0])

        # Update cost and suggested price
        recipe.cost_per_unit = predicted_cost
        if recipe.profit_margin:
            recipe.suggested_price = predicted_cost * (1 + recipe.profit_margin / 100)

        db.commit()
        db.refresh(recipe)

        return {
            "id": recipe.id,
            "recipe_name": recipe.recipe_name,
            "recipe_category": recipe.recipe_category,
            "cost_per_unit": recipe.cost_per_unit,
            "profit_margin": recipe.profit_margin,
            "suggested_price": recipe.suggested_price,
            "ingredient_cost": recipe.ingredient_cost,
            "quantity": recipe.quantity,
            "waste_percentage": recipe.waste_percentage,
            "processing_cost": recipe.processing_cost,
            "labor_hours": recipe.labor_hours,
            "labor_rate": recipe.labor_rate,
            "overhead_cost": recipe.overhead_cost,
            "packaging_cost": recipe.packaging_cost,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating recipe: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=Dict[str, Any])
def get_model_info():
    """Get information about the current model"""
    try:
        return model_manager.current_model_info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/retrain", response_model=Dict[str, Any])
def retrain_model(db: Session = Depends(get_db)):
    """Manually trigger model retraining"""
    try:
        model, model_info = train_models(db)
        return {"message": "Model retrained successfully", "model_info": model_info}
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", response_model=List[str])
def get_categories(db: Session = Depends(get_db)):
    """Get list of all recipe categories"""
    try:
        categories = (
            db.query(RecipeCost.recipe_category)
            .filter(RecipeCost.recipe_category != None)
            .distinct()
            .all()
        )
        return [category[0] for category in categories if category[0]]
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", response_model=Dict[str, Any])
def get_statistics(db: Session = Depends(get_db)):
    """Get statistical information about recipes"""
    try:
        # Count recipes
        total_recipes = db.query(func.count(RecipeCost.id)).scalar()

        # Get average cost
        avg_cost = db.query(func.avg(RecipeCost.cost_per_unit)).scalar()

        # Get min/max costs
        min_cost = db.query(func.min(RecipeCost.cost_per_unit)).scalar()
        max_cost = db.query(func.max(RecipeCost.cost_per_unit)).scalar()

        # Count by category
        category_counts = (
            db.query(RecipeCost.recipe_category, func.count(RecipeCost.id))
            .group_by(RecipeCost.recipe_category)
            .all()
        )

        category_stats = {}
        for category, count in category_counts:
            if category:
                category_stats[category] = count

        return {
            "total_recipes": total_recipes,
            "average_cost": float(avg_cost) if avg_cost else 0,
            "min_cost": float(min_cost) if min_cost else 0,
            "max_cost": float(max_cost) if max_cost else 0,
            "categories": category_stats,
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to get training data from DB - was missing in the original code
def get_training_data():
    try:
        session = SessionLocal()
        data = session.query(RecipeCost).all()
        session.close()

        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for d in data:
            record = {
                "ingredient_cost": d.ingredient_cost,
                "quantity": d.quantity,
                "waste_percentage": d.waste_percentage,
                "processing_cost": d.processing_cost,
                "cost_per_unit": d.cost_per_unit,
            }

            # Add optional fields if they exist
            if hasattr(d, "labor_hours") and d.labor_hours is not None:
                record["labor_hours"] = d.labor_hours
            if hasattr(d, "labor_rate") and d.labor_rate is not None:
                record["labor_rate"] = d.labor_rate
            if hasattr(d, "overhead_cost") and d.overhead_cost is not None:
                record["overhead_cost"] = d.overhead_cost
            if hasattr(d, "packaging_cost") and d.packaging_cost is not None:
                record["packaging_cost"] = d.packaging_cost

            records.append(record)

        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Error getting training data: {str(e)}")
        return pd.DataFrame()


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
