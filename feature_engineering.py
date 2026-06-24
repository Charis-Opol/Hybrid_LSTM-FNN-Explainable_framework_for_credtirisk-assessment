"""Localised feature engineering for Ugandan microfinance credit risk.

The functions in this module derive monthly borrower features from transaction
histories. They are intentionally transparent: each feature has a formula,
credit-risk motivation, and Ugandan relevance documented for research use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BORROWER_ID_COLUMN, TARGET_COLUMN, TRANSACTION_DATE_COLUMN


LOGGER = logging.getLogger(__name__)
EPSILON = 1e-9


@dataclass(frozen=True)
class FeatureColumns:
    """Column names used by the feature engineering pipeline."""

    borrower_id: str = BORROWER_ID_COLUMN
    transaction_date: str = TRANSACTION_DATE_COLUMN
    amount: str = "transaction_amount"
    transaction_type: str = "transaction_type"
    balance: str = "balance"
    target: str = TARGET_COLUMN


class LocalisedFeatureEngineer:
    """Engineer financial, behavioural, dormancy, and seasonal features."""

    def __init__(
        self,
        columns: FeatureColumns | None = None,
        analysis_date: str | pd.Timestamp | None = None,
    ) -> None:
        self.columns = columns or FeatureColumns()
        self.analysis_date = (
            pd.Timestamp(analysis_date) if analysis_date is not None else None
        )

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a final engineered monthly dataframe.

        Args:
            data: Transaction-level dataframe.

        Returns:
            Borrower-month dataframe with localised features and behavioural
            scores.
        """

        frame = self._prepare_transactions(data)
        monthly = self._monthly_financial_features(frame)
        monthly = self._behaviour_features(frame, monthly)
        monthly = self._dormancy_features(frame, monthly)
        monthly = self._seasonal_features(monthly)
        monthly = self._behavioural_scores(monthly)

        if self.columns.target in frame:
            labels = frame.groupby(self.columns.borrower_id)[self.columns.target].max()
            monthly[self.columns.target] = monthly[self.columns.borrower_id].map(labels)

        LOGGER.info("Engineered feature dataframe shape: %s", monthly.shape)
        return monthly.sort_values(
            [self.columns.borrower_id, "month"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _prepare_transactions(self, data: pd.DataFrame) -> pd.DataFrame:
        required = [
            self.columns.borrower_id,
            self.columns.transaction_date,
            self.columns.amount,
        ]
        missing = [column for column in required if column not in data]
        if missing:
            raise ValueError("Missing required feature columns: " + ", ".join(missing))

        frame = data.copy()
        frame[self.columns.transaction_date] = pd.to_datetime(
            frame[self.columns.transaction_date],
            errors="coerce",
        )
        frame = frame.dropna(
            subset=[
                self.columns.borrower_id,
                self.columns.transaction_date,
                self.columns.amount,
            ]
        )
        frame[self.columns.amount] = pd.to_numeric(
            frame[self.columns.amount],
            errors="coerce",
        ).fillna(0)
        frame["month"] = frame[self.columns.transaction_date].dt.to_period("M")
        frame["signed_amount"] = self._signed_amount(frame)
        frame["inflow_amount"] = frame["signed_amount"].clip(lower=0)
        frame["outflow_amount"] = (-frame["signed_amount"].clip(upper=0))

        if self.columns.balance in frame:
            frame[self.columns.balance] = pd.to_numeric(
                frame[self.columns.balance],
                errors="coerce",
            )
        else:
            frame[self.columns.balance] = np.nan

        return frame

    def _signed_amount(self, frame: pd.DataFrame) -> pd.Series:
        if self.columns.transaction_type not in frame:
            return frame[self.columns.amount]

        transaction_type = (
            frame[self.columns.transaction_type].astype(str).str.lower().str.strip()
        )
        sent_tokens = ["sent", "withdrawal", "debit", "payment", "outflow", "paid"]
        received_tokens = ["received", "deposit", "credit", "inflow", "income"]

        amount = frame[self.columns.amount].abs()
        signed = frame[self.columns.amount].copy()
        signed[transaction_type.str.contains("|".join(sent_tokens), na=False)] = -amount
        signed[transaction_type.str.contains("|".join(received_tokens), na=False)] = (
            amount
        )
        return signed

    def _monthly_financial_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby([self.columns.borrower_id, "month"])
        monthly = grouped.agg(
            monthly_inflow=("inflow_amount", "sum"),
            monthly_outflow=("outflow_amount", "sum"),
            average_balance=(self.columns.balance, "mean"),
            balance_volatility=(self.columns.balance, "std"),
            average_transaction=(self.columns.amount, "mean"),
            median_transaction=(self.columns.amount, "median"),
            transaction_count=(self.columns.amount, "size"),
        ).reset_index()
        monthly["balance_volatility"] = monthly["balance_volatility"].fillna(0)
        monthly["average_balance"] = monthly["average_balance"].fillna(0)
        return monthly.assign(
            net_cashflow=self.net_cashflow(
                monthly["monthly_inflow"],
                monthly["monthly_outflow"],
            )
        )

    @staticmethod
    def monthly_inflow(inflow: pd.Series) -> pd.Series:
        """Formula: sum of incoming transaction values per borrower-month.

        Credit-risk relevance: stable inflows suggest repayment capacity.
        Uganda relevance: many microfinance borrowers receive seasonal, mobile
        money, farming, or trading income that must be measured monthly.
        """

        return inflow

    @staticmethod
    def monthly_outflow(outflow: pd.Series) -> pd.Series:
        """Formula: sum of outgoing transaction values per borrower-month.

        Credit-risk relevance: high outflows can reduce disposable cash.
        Uganda relevance: household, school fees, farming input, and remittance
        expenses often cluster around local calendar events.
        """

        return outflow

    @staticmethod
    def net_cashflow(inflow: pd.Series, outflow: pd.Series) -> pd.Series:
        """Formula: monthly inflow minus monthly outflow.

        Credit-risk relevance: positive net cashflow indicates repayment room.
        Uganda relevance: informal and agricultural borrowers may be viable
        even with volatile income if net cashflow is positive after local costs.
        """

        return inflow - outflow

    @staticmethod
    def average_balance(balance: pd.Series) -> float:
        """Formula: arithmetic mean of balances in a borrower-month.

        Credit-risk relevance: higher balances provide a repayment buffer.
        Uganda relevance: SACCO and mobile wallet balances can reveal liquidity
        not captured by formal payslips.
        """

        return float(balance.mean())

    @staticmethod
    def balance_volatility(balance: pd.Series) -> float:
        """Formula: standard deviation of balances in a borrower-month.

        Credit-risk relevance: unstable balances may indicate liquidity stress.
        Uganda relevance: traders and farmers often experience irregular cash
        cycles, so volatility helps separate rhythm from fragility.
        """

        return float(balance.std())

    @staticmethod
    def average_transaction(amount: pd.Series) -> float:
        """Formula: arithmetic mean of transaction values in a month.

        Credit-risk relevance: transaction size approximates financial scale.
        Uganda relevance: microenterprise and mobile money activity is often
        better observed through transaction size than formal income records.
        """

        return float(amount.mean())

    @staticmethod
    def median_transaction(amount: pd.Series) -> float:
        """Formula: median transaction value in a borrower-month.

        Credit-risk relevance: the median resists extreme one-off payments.
        Uganda relevance: occasional bulk harvest payments can distort averages,
        so median size gives a steadier local affordability signal.
        """

        return float(amount.median())

    def _behaviour_features(
        self,
        frame: pd.DataFrame,
        monthly: pd.DataFrame,
    ) -> pd.DataFrame:
        days_per_month = monthly["month"].dt.days_in_month
        monthly["transaction_velocity"] = self.transaction_velocity(
            monthly["transaction_count"],
            days_per_month,
        )
        monthly["transaction_frequency"] = self.transaction_frequency(
            monthly["transaction_count"]
        )
        monthly["sent_received_ratio"] = self.sent_received_ratio(
            monthly["monthly_outflow"],
            monthly["monthly_inflow"],
        )
        monthly["savings_ratio"] = self.savings_ratio(
            monthly["net_cashflow"],
            monthly["monthly_inflow"],
        )
        monthly["expenditure_ratio"] = self.expenditure_ratio(
            monthly["monthly_outflow"],
            monthly["monthly_inflow"],
        )
        monthly["liquidity_score"] = self.liquidity_score(
            monthly["average_balance"],
            monthly["monthly_outflow"],
        )

        borrower_stats = monthly.groupby(self.columns.borrower_id).agg(
            inflow_mean=("monthly_inflow", "mean"),
            inflow_std=("monthly_inflow", "std"),
            cashflow_mean=("net_cashflow", "mean"),
            cashflow_std=("net_cashflow", "std"),
        )
        monthly["income_regularity"] = monthly[self.columns.borrower_id].map(
            self.income_regularity(
                borrower_stats["inflow_mean"],
                borrower_stats["inflow_std"].fillna(0),
            )
        )
        monthly["cashflow_stability"] = monthly[self.columns.borrower_id].map(
            self.cashflow_stability(
                borrower_stats["cashflow_mean"],
                borrower_stats["cashflow_std"].fillna(0),
            )
        )
        return monthly

    @staticmethod
    def transaction_velocity(count: pd.Series, days: pd.Series) -> pd.Series:
        """Formula: transaction count divided by days in the month.

        Credit-risk relevance: active accounts provide richer repayment signals.
        Uganda relevance: mobile money usage intensity can reveal business
        activity where formal bookkeeping is limited.
        """

        return count / days.replace(0, np.nan)

    @staticmethod
    def transaction_frequency(count: pd.Series) -> pd.Series:
        """Formula: number of transactions in a borrower-month.

        Credit-risk relevance: frequent use suggests observable, active cashflow.
        Uganda relevance: informal borrowers may leave their strongest data
        trail through repeated small mobile or SACCO transactions.
        """

        return count

    @staticmethod
    def sent_received_ratio(outflow: pd.Series, inflow: pd.Series) -> pd.Series:
        """Formula: monthly outflow divided by monthly inflow.

        Credit-risk relevance: ratios above one imply spending exceeds receipts.
        Uganda relevance: remittance and merchant payments can reveal pressure
        around family obligations, inputs, rent, and local business expenses.
        """

        return outflow / (inflow + EPSILON)

    @staticmethod
    def income_regularity(mean: pd.Series, std: pd.Series) -> pd.Series:
        """Formula: 1 divided by 1 plus coefficient of variation of inflow.

        Credit-risk relevance: regular income improves repayment predictability.
        Uganda relevance: it distinguishes salaried, trading, and agricultural
        income patterns common in local microfinance portfolios.
        """

        return 1 / (1 + (std / (mean.abs() + EPSILON)))

    @staticmethod
    def liquidity_score(balance: pd.Series, outflow: pd.Series) -> pd.Series:
        """Formula: average balance divided by monthly outflow plus epsilon.

        Credit-risk relevance: liquidity cushions repayment shocks.
        Uganda relevance: balances in wallets, SACCOs, and agent banking
        channels can signal short-term resilience.
        """

        return balance / (outflow + EPSILON)

    @staticmethod
    def cashflow_stability(mean: pd.Series, std: pd.Series) -> pd.Series:
        """Formula: 1 divided by 1 plus coefficient of variation of cashflow.

        Credit-risk relevance: stable net cashflow supports predictable payment.
        Uganda relevance: this captures whether seasonal cash movements still
        follow a reliable borrower-specific rhythm.
        """

        return 1 / (1 + (std / (mean.abs() + EPSILON)))

    @staticmethod
    def savings_ratio(net_cashflow: pd.Series, inflow: pd.Series) -> pd.Series:
        """Formula: positive net cashflow divided by monthly inflow.

        Credit-risk relevance: retained surplus indicates repayment capacity.
        Uganda relevance: it approximates savings discipline where formal bank
        savings records may be incomplete.
        """

        return net_cashflow.clip(lower=0) / (inflow + EPSILON)

    @staticmethod
    def expenditure_ratio(outflow: pd.Series, inflow: pd.Series) -> pd.Series:
        """Formula: monthly outflow divided by monthly inflow.

        Credit-risk relevance: high expenditure share reduces loan affordability.
        Uganda relevance: school fees, farming inputs, and festive obligations
        can create predictable local expenditure peaks.
        """

        return outflow / (inflow + EPSILON)

    def _dormancy_features(
        self,
        frame: pd.DataFrame,
        monthly: pd.DataFrame,
    ) -> pd.DataFrame:
        dormancy = frame.sort_values(
            [self.columns.borrower_id, self.columns.transaction_date]
        ).copy()
        dormancy["gap_days"] = dormancy.groupby(self.columns.borrower_id)[
            self.columns.transaction_date
        ].diff().dt.days

        latest_date = (
            self.analysis_date
            if self.analysis_date is not None
            else frame[self.columns.transaction_date].max()
        )
        by_borrower = dormancy.groupby(self.columns.borrower_id).agg(
            last_transaction=(self.columns.transaction_date, "max"),
            maximum_dormancy=("gap_days", "max"),
            average_dormancy=("gap_days", "mean"),
        )
        by_borrower["days_inactive"] = (
            latest_date - by_borrower["last_transaction"]
        ).dt.days
        by_borrower = by_borrower.fillna(0)

        recovery = self.recovery_after_dormancy(
            dormancy,
            self.columns.borrower_id,
        )
        by_borrower["recovery_after_dormancy"] = recovery.reindex(
            by_borrower.index,
            fill_value=0,
        )

        for column in [
            "days_inactive",
            "maximum_dormancy",
            "average_dormancy",
            "recovery_after_dormancy",
        ]:
            monthly[column] = monthly[self.columns.borrower_id].map(
                by_borrower[column]
            )
        return monthly

    @staticmethod
    def days_inactive(latest_date: pd.Timestamp, last_date: pd.Series) -> pd.Series:
        """Formula: analysis date minus borrower's latest transaction date.

        Credit-risk relevance: recent inactivity can signal distress or churn.
        Uganda relevance: inactivity may reflect seasonal rural income cycles,
        network changes, or reduced mobile money activity.
        """

        return (latest_date - last_date).dt.days

    @staticmethod
    def maximum_dormancy(gaps: pd.Series) -> float:
        """Formula: maximum days between consecutive borrower transactions.

        Credit-risk relevance: long gaps reduce confidence in cashflow evidence.
        Uganda relevance: harvest timing, travel, and agent liquidity can create
        meaningful pauses in financial activity.
        """

        return float(gaps.max())

    @staticmethod
    def average_dormancy(gaps: pd.Series) -> float:
        """Formula: mean days between consecutive borrower transactions.

        Credit-risk relevance: shorter average gaps imply consistent activity.
        Uganda relevance: regular small transactions are common in mobile money
        ecosystems and can proxy business continuity.
        """

        return float(gaps.mean())

    @staticmethod
    def recovery_after_dormancy(
        frame: pd.DataFrame,
        borrower_id_column: str = BORROWER_ID_COLUMN,
    ) -> pd.Series:
        """Formula: average signed amount after gaps of at least 30 days.

        Credit-risk relevance: recovery after inactivity shows resilience.
        Uganda relevance: farmers and traders may pause before harvest or market
        cycles, so post-gap recovery is more useful than dormancy alone.
        """

        recovered = frame[frame["gap_days"] >= 30]
        if recovered.empty:
            return pd.Series(0, index=frame[borrower_id_column].unique())
        return recovered.groupby(borrower_id_column)["signed_amount"].mean()

    def _seasonal_features(self, monthly: pd.DataFrame) -> pd.DataFrame:
        month_number = monthly["month"].dt.month
        monthly["harvest_season"] = self.harvest_season(month_number)
        monthly["school_fees_season"] = self.school_fees_season(month_number)
        monthly["christmas_season"] = self.christmas_season(month_number)
        monthly["january_recovery"] = self.january_recovery(month_number)
        monthly["rainy_season"] = self.rainy_season(month_number)
        monthly["dry_season"] = self.dry_season(month_number)
        monthly["agricultural_income_period"] = self.agricultural_income_period(
            month_number
        )
        return monthly

    @staticmethod
    def harvest_season(month: pd.Series) -> pd.Series:
        """Formula: binary flag for common harvest months June, July, November,
        and December.

        Credit-risk relevance: harvest income can temporarily improve repayment.
        Uganda relevance: many borrowers depend on crop cycles, so harvest
        months can explain inflow spikes.
        """

        return month.isin([6, 7, 11, 12]).astype(int)

    @staticmethod
    def school_fees_season(month: pd.Series) -> pd.Series:
        """Formula: binary flag for common school-fee months January, February,
        May, and September.

        Credit-risk relevance: school fees can compete with loan installments.
        Uganda relevance: household education expenses are a major recurring
        local cashflow pressure.
        """

        return month.isin([1, 2, 5, 9]).astype(int)

    @staticmethod
    def christmas_season(month: pd.Series) -> pd.Series:
        """Formula: binary flag equal to one in December.

        Credit-risk relevance: festive spending may raise short-term default risk.
        Uganda relevance: December travel, food, gifts, and family obligations
        often increase spending.
        """

        return (month == 12).astype(int)

    @staticmethod
    def january_recovery(month: pd.Series) -> pd.Series:
        """Formula: binary flag equal to one in January.

        Credit-risk relevance: January may show recovery or strain after holiday
        spending.
        Uganda relevance: borrowers often face school fees and post-Christmas
        liquidity rebuilding in January.
        """

        return (month == 1).astype(int)

    @staticmethod
    def rainy_season(month: pd.Series) -> pd.Series:
        """Formula: binary flag for March-May and September-November.

        Credit-risk relevance: rains can affect sales, farming costs, and income.
        Uganda relevance: bimodal rainfall patterns influence agriculture,
        transport, and trading activity.
        """

        return month.isin([3, 4, 5, 9, 10, 11]).astype(int)

    @staticmethod
    def dry_season(month: pd.Series) -> pd.Series:
        """Formula: binary flag for December-February and June-August.

        Credit-risk relevance: dry months can shift income and expenses.
        Uganda relevance: dry seasons affect crop sales, water costs, transport,
        and market access in many districts.
        """

        return month.isin([12, 1, 2, 6, 7, 8]).astype(int)

    @staticmethod
    def agricultural_income_period(month: pd.Series) -> pd.Series:
        """Formula: binary flag for expected farm income months June, July,
        November, and December.

        Credit-risk relevance: farm income periods can improve repayment ability.
        Uganda relevance: agriculture is central to many microfinance borrowers,
        making seasonal income timing locally important.
        """

        return month.isin([6, 7, 11, 12]).astype(int)

    def _behavioural_scores(self, monthly: pd.DataFrame) -> pd.DataFrame:
        normalized = self._min_max_normalize(
            monthly[
                [
                    "income_regularity",
                    "cashflow_stability",
                    "liquidity_score",
                    "savings_ratio",
                    "transaction_frequency",
                    "sent_received_ratio",
                    "expenditure_ratio",
                    "balance_volatility",
                    "net_cashflow",
                    "average_balance",
                ]
            ]
        )

        monthly["financial_stability_score"] = self.financial_stability_score(
            normalized
        )
        monthly["behaviour_score"] = self.behaviour_score(normalized)
        monthly["cashflow_consistency"] = normalized["cashflow_stability"]
        monthly["income_consistency"] = normalized["income_regularity"]
        monthly["balance_growth"] = monthly.groupby(self.columns.borrower_id)[
            "average_balance"
        ].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        monthly["activity_score"] = normalized["transaction_frequency"]
        monthly["credit_behaviour_score"] = self.credit_behaviour_score(
            monthly,
            normalized,
        )
        monthly["spending_score"] = self.spending_score(normalized)
        return monthly

    @staticmethod
    def financial_stability_score(normalized: pd.DataFrame) -> pd.Series:
        """Formula: weighted sum of liquidity, cashflow stability, savings, and
        net cashflow after min-max normalization.

        Credit-risk relevance: combines repayment buffer and cashflow quality.
        Uganda relevance: it balances seasonal income with liquidity signals
        common in SACCO and mobile money data.
        """

        return (
            0.30 * normalized["liquidity_score"]
            + 0.30 * normalized["cashflow_stability"]
            + 0.20 * normalized["savings_ratio"]
            + 0.20 * normalized["net_cashflow"]
        )

    @staticmethod
    def behaviour_score(normalized: pd.DataFrame) -> pd.Series:
        """Formula: weighted sum of activity, income regularity, and inverse
        sent/received pressure after normalization.

        Credit-risk relevance: rewards active, regular, balanced behaviour.
        Uganda relevance: reflects mobile money and informal cashflow patterns
        without requiring formal credit bureau depth.
        """

        return (
            0.40 * normalized["transaction_frequency"]
            + 0.35 * normalized["income_regularity"]
            + 0.25 * (1 - normalized["sent_received_ratio"])
        )

    @staticmethod
    def credit_behaviour_score(
        monthly: pd.DataFrame,
        normalized: pd.DataFrame,
    ) -> pd.Series:
        """Formula: weighted blend of financial stability, behaviour, and low
        inactivity risk.

        Credit-risk relevance: summarizes repayment readiness into one score.
        Uganda relevance: combines localized cash cycles, transaction activity,
        and dormancy patterns for loan-officer interpretation.
        """

        inactivity = LocalisedFeatureEngineer._min_max_normalize(
            monthly[["days_inactive"]]
        )["days_inactive"]
        return (
            0.45 * monthly["financial_stability_score"]
            + 0.40 * monthly["behaviour_score"]
            + 0.15 * (1 - inactivity)
        )

    @staticmethod
    def spending_score(normalized: pd.DataFrame) -> pd.Series:
        """Formula: one minus normalized expenditure ratio and balance volatility,
        averaged with inverse sent/received pressure.

        Credit-risk relevance: lower spending pressure supports affordability.
        Uganda relevance: controls for school fees, holiday, and farm-input
        expense cycles that affect repayment timing.
        """

        return (
            (1 - normalized["expenditure_ratio"])
            + (1 - normalized["balance_volatility"])
            + (1 - normalized["sent_received_ratio"])
        ) / 3

    @staticmethod
    def _min_max_normalize(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        minimum = numeric.min()
        maximum = numeric.max()
        denominator = (maximum - minimum).replace(0, 1)
        return (numeric - minimum) / denominator


def engineer_features(
    data: pd.DataFrame,
    columns: FeatureColumns | None = None,
    analysis_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Engineer all localised borrower-month features.

    Args:
        data: Transaction-level dataframe.
        columns: Optional custom input column names.
        analysis_date: Optional date for inactivity calculations.

    Returns:
        Engineered borrower-month dataframe.
    """

    return LocalisedFeatureEngineer(columns=columns, analysis_date=analysis_date).engineer(
        data
    )
