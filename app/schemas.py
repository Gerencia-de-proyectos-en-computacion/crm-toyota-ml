from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    part_id: str = Field(min_length=1)
    vehicle_model: str = Field(min_length=1)
    part_category: str = Field(min_length=1)
    avg_mileage: float = Field(ge=0, description="Kilometraje promedio de la semana anterior")
    vehicles_served: int = Field(ge=0, description="Vehiculos atendidos la semana anterior")
    month: int = Field(ge=1, le=12)
    week_of_year: int = Field(ge=1, le=53)
    lag_1: float = Field(ge=0)
    lag_2: float = Field(ge=0)
    lag_4: float = Field(ge=0)
    rolling_mean_4: float = Field(ge=0)
    rolling_std_4: float = Field(ge=0, description="Desviacion poblacional (ddof=0)")


class BatchRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    part_id: str
    predicted_demand: float
    reorder_point: int
    horizon_days: int = 7
    model_version: str
