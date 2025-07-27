package models

import (
	"errors"
	"strings"
	"time"

	"github.com/go-playground/validator/v10"
)

// Use a single validator instance (recommended for performance)
var validate *validator.Validate

func init() {
	validate = validator.New()

	// Register custom validation functions
	validate.RegisterValidation("notblank", validateNotBlank)
}

// Todo represents a single todo item
type Todo struct {
	ID          int       `json:"id" db:"id"`
	Title       string    `json:"title" db:"title"`
	Description string    `json:"description" db:"description"`
	Completed   bool      `json:"completed" db:"completed"`
	CreatedAt   time.Time `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time `json:"updated_at" db:"updated_at"`
}

// TodoRequest represents the expected JSON structure for creating/updating todos
type TodoRequest struct {
	Title       string `json:"title" binding:"required" validate:"min=1,max=255"`
	Description string `json:"description" validate:"max=1000"`
	Completed   bool   `json:"completed"`
}

// TodoFilter represents the expected query parameters for filtering todos
type TodoFilter struct {
	Completed *bool  `form:"completed"`
	Search    string `form:"search"`
	Limit     int    `form:"limit,default=10" validate:"min=1,max=100"`
	Offset    int    `form:"offset,default=0" validate:"min=0"`
}

// Validate performs comprehensive validation on TodoRequest
func (tr *TodoRequest) Validate() error {
	// Use the validator package for struct validation
	if err := validate.Struct(tr); err != nil {
		// Convert validator errors to user-friendly messages
		return formatValidationError(err)
	}

	// Additional custom validation
	if len(strings.TrimSpace(tr.Title)) == 0 {
		return errors.New("title cannot be empty or only whitespace")
	}

	return nil
}

// Custom validation function for non-blank strings
func validateNotBlank(fl validator.FieldLevel) bool {
	return len(strings.TrimSpace(fl.Field().String())) > 0
}

// formatValidationError converts validator errors to user-friendly messages
func formatValidationError(err error) error {
	var validationErrors []string

	for _, err := range err.(validator.ValidationErrors) {
		switch err.Tag() {
		case "required":
			validationErrors = append(validationErrors, err.Field()+" is required")
		case "min":
			validationErrors = append(validationErrors, err.Field()+" must be at least "+err.Param()+" characters")
		case "max":
			validationErrors = append(validationErrors, err.Field()+" must be at most "+err.Param()+" characters")
		case "notblank":
			validationErrors = append(validationErrors, err.Field()+" cannot be blank")
		default:
			validationErrors = append(validationErrors, err.Field()+" is invalid")
		}
	}

	return errors.New(strings.Join(validationErrors, ", "))
}

// Sanitize cleans up the todo request data
func (tr *TodoRequest) Sanitize() {
	tr.Title = strings.TrimSpace(tr.Title)
	tr.Description = strings.TrimSpace(tr.Description)
}
