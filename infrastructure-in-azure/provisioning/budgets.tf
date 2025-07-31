resource "azurerm_consumption_budget_subscription" "monthly_budget" {
  name            = "monthly-budget"
  subscription_id = "/subscriptions/${data.azurerm_subscription.primary.subscription_id}"
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
	start_date = "${formatdate("YYYY-MM-01", timestamp())}T00:00:00Z"
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = var.budget_threshold_percentage
    threshold_type = "Actual"
    
    contact_emails = var.alert_emails
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Forecasted"
    
    contact_emails = var.alert_emails
  }
}