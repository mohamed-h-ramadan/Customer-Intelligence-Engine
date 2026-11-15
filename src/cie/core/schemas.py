"""Data schemas and Pydantic models for customer features, regime validation, and scoring."""

from pydantic import BaseModel, Field, model_validator


class CustomerFeatures(BaseModel):
    """Features required to score a single customer.

    Represents the 26 features computed strictly over a 270-day lookback window.
    """

    customer_id: str | int | None = Field(
        default=None,
        description="Unique registered customer identifier (e.g. 12347).",
        examples=[12347],
    )
    recency_days: float = Field(
        ...,
        ge=0.0,
        description="Days since customer's most recent invoice as of cutoff.",
        examples=[1.9],
    )
    frequency: float = Field(
        ...,
        ge=1.0,
        description="Count of distinct sales invoices in the lookback window.",
        examples=[5.0],
    )
    monetary: float = Field(
        ...,
        ge=0.0,
        description="Total gross sales spend (£) in the lookback window.",
        examples=[3100.50],
    )
    aov: float = Field(
        ...,
        ge=0.0,
        description="Average order value (£) (monetary / frequency).",
        examples=[620.10],
    )
    avg_items_per_invoice: float = Field(
        ...,
        ge=0.0,
        description="Mean quantity of items purchased per invoice.",
        examples=[185.2],
    )
    avg_unit_price: float = Field(
        ...,
        ge=0.0,
        description="Quantity-weighted mean item price (£).",
        examples=[3.35],
    )
    n_unique_products: float = Field(
        ...,
        ge=1.0,
        description="Number of distinct StockCodes purchased.",
        examples=[45.0],
    )
    n_active_months: float = Field(
        ...,
        ge=1.0,
        le=12.0,
        description="Count of distinct calendar months with purchase activity.",
        examples=[4.0],
    )
    n_active_weeks: float = Field(
        ...,
        ge=1.0,
        le=52.0,
        description="Count of distinct calendar weeks with purchase activity.",
        examples=[5.0],
    )
    tenure_days_in_window: float = Field(
        ...,
        ge=0.0,
        description="Days between first and last purchase in the lookback window.",
        examples=[210.5],
    )
    is_tenure_at_edge: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag (1/0) indicating whether first purchase was within 7d of lookback boundary.",
        examples=[0],
    )
    mean_interpurchase_days: float | None = Field(
        default=None,
        description="Mean days between consecutive purchases. Structurally null for single-invoice buyers.",
        examples=[52.6],
    )
    is_single_invoice: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag (1/0) indicating whether customer had exactly 1 invoice in lookback.",
        examples=[0],
    )
    spend_last_30d: float = Field(
        ...,
        ge=0.0,
        description="Gross spend (£) in the 30 days immediately preceding cutoff.",
        examples=[580.0],
    )
    spend_last_60d: float = Field(
        ...,
        ge=0.0,
        description="Gross spend (£) in the 60 days immediately preceding cutoff.",
        examples=[1200.0],
    )
    spend_last_90d: float = Field(
        ...,
        ge=0.0,
        description="Gross spend (£) in the 90 days immediately preceding cutoff.",
        examples=[1650.0],
    )
    invoices_last_30d: float = Field(
        ...,
        ge=0.0,
        description="Count of invoices in the last 30 days.",
        examples=[1.0],
    )
    invoices_last_60d: float = Field(
        ...,
        ge=0.0,
        description="Count of invoices in the last 60 days.",
        examples=[2.0],
    )
    invoices_last_90d: float = Field(
        ...,
        ge=0.0,
        description="Count of invoices in the last 90 days.",
        examples=[3.0],
    )
    spend_last_90d_ratio: float = Field(
        ...,
        ge=0.0,
        description="Share of total lookback spend occurring in the last 90 days.",
        examples=[0.53],
    )
    return_invoice_count: float = Field(
        ...,
        ge=0.0,
        description="Count of distinct return/credit invoices in lookback.",
        examples=[0.0],
    )
    return_value_abs: float = Field(
        ...,
        ge=0.0,
        description="Absolute gross value (£) of goods returned in lookback.",
        examples=[0.0],
    )
    return_value_ratio: float = Field(
        ...,
        ge=0.0,
        description="Ratio of returned value to gross sales spend.",
        examples=[0.0],
    )
    return_rate: float = Field(
        ...,
        ge=0.0,
        description="Ratio of return invoice count to total invoices.",
        examples=[0.0],
    )
    is_uk: int = Field(
        ...,
        ge=0,
        le=1,
        description="Flag (1/0) indicating whether customer's primary billing country is United Kingdom.",
        examples=[1],
    )
    country: str = Field(
        ...,
        description="Customer's country name as recorded in the dataset.",
        examples=["United Kingdom"],
    )

    @model_validator(mode="after")
    def validate_structural_nulls(self) -> "CustomerFeatures":
        """Assert structural NaN contract: mean_interpurchase_days is null iff is_single_invoice == 1."""
        if self.is_single_invoice == 1:
            if self.mean_interpurchase_days is not None and not (
                self.mean_interpurchase_days != self.mean_interpurchase_days
            ):  # checking float NaN
                # If provided for single invoice, set to None/NaN as structural contract
                pass
        else:
            if self.mean_interpurchase_days is None:
                raise ValueError(
                    "mean_interpurchase_days cannot be null for repeat customers (is_single_invoice == 0)."
                )
        return self


class ChallengersResponse(BaseModel):
    """Predictions from designated benchmark / challenger models."""

    run_rate: float = Field(
        ...,
        description="Historical run-rate heuristic: monetary * (90 / 270) (£).",
        examples=[1033.50],
    )
    calibrated_run_rate: float = Field(
        ...,
        description="Seasonally calibrated run-rate (£) applying empirical Q4 expansion factor (1.427x).",
        examples=[1475.28],
    )
    train_mean: float = Field(
        ...,
        description="Training set mean spend (£752.81).",
        examples=[752.81],
    )
    train_median: float = Field(
        ...,
        description="Training set median spend (£222.08).",
        examples=[222.08],
    )


class CustomerScoringResponse(BaseModel):
    """Complete scoring response for a single customer."""

    customer_id: str | int | None = Field(
        default=None,
        description="Customer identifier.",
        examples=[12347],
    )
    expected_clv: float = Field(
        ...,
        description="Champion Tweedie GLM predicted 90-day forward gross spend (£).",
        examples=[1450.85],
    )
    challengers: ChallengersResponse = Field(
        ...,
        description="Predictions from challenger and baseline models for shipping evaluation.",
    )
    segment: str = Field(
        ...,
        description="Unsupervised CRM segment label (for marketing routing only; never a CLV feature).",
        examples=["VIP Champions"],
    )
    crm_action: str = Field(
        ...,
        description="Recommended operational CRM playbook for this segment.",
        examples=["Dedicated B2B account rep, priority inventory allocation, bespoke tier pricing"],
    )
    model_version: str = Field(
        ...,
        description="Frozen champion model artifact version tag.",
        examples=["v1.0-frozen-2010-09-10"],
    )
    p_active: float | None = Field(
        default=None,
        description="Stage-1 propensity of active repeat purchase (if hurdle fallback is activated).",
        examples=[0.925],
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of explicit warnings if inputs breach validated regime boundaries.",
        examples=[],
    )


class BatchCustomerScoringResponse(BaseModel):
    """Response schema for batch scoring requests."""

    total_customers: int = Field(..., description="Number of customers scored.", examples=[100])
    scored_customers: list[CustomerScoringResponse] = Field(
        ..., description="List of individual scoring responses."
    )
    model_version: str = Field(..., description="Champion model version tag.")
    total_expected_clv: float = Field(
        ..., description="Sum of expected gross CLV across all scored customers (£)."
    )
    warnings_count: int = Field(
        ..., description="Total count of customer records that triggered regime warnings."
    )
