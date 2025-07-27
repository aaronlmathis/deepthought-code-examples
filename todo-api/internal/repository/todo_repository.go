package repository

import (
	"database/sql"
	"fmt"
	"strings"

	"todo-api/internal/database"
	"todo-api/internal/models"
)

type TodoRepository struct {
	db database.TodoDB
}

// NewTodoRepository creates a new TodoRepository with the provided database interface
// Using an interface instead of a concrete type makes unit testing easier by allowing
// us to inject mock implementations without requiring a real database connection
func NewTodoRepository(db database.TodoDB) *TodoRepository {
	return &TodoRepository{db: db}
}

// GetAll retrieves all todo items with optional filtering.
func (r *TodoRepository) GetAll(filter models.TodoFilter) ([]*models.Todo, error) {
	query := "SELECT id, title, description, completed, created_at, updated_at FROM todos"
	args := []interface{}{}
	conditions := []string{}
	argIndex := 1

	if filter.Completed != nil {
		conditions = append(conditions, fmt.Sprintf("completed = $%d", argIndex))
		args = append(args, *filter.Completed)
		argIndex++
	}

	if filter.Search != "" {
		searchPattern := "%" + filter.Search + "%"
		conditions = append(conditions, fmt.Sprintf("(title ILIKE $%d OR description ILIKE $%d)", argIndex, argIndex))
		args = append(args, searchPattern)
		argIndex++
	}

	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	query += " ORDER BY created_at DESC"
	query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, filter.Limit, filter.Offset)

	todos := []*models.Todo{}
	err := r.db.Select(&todos, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to get todos: %w", err)
	}

	return todos, nil
}

// GetByID retrieves a todo item by its ID.
func (r *TodoRepository) GetByID(id int) (*models.Todo, error) {
	todo := &models.Todo{}
	query := "SELECT id, title, description, completed, created_at, updated_at FROM todos WHERE id = $1"

	err := r.db.Get(todo, query, id)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("todo with id %d not found", id)
		}
		return nil, fmt.Errorf("failed to get todo: %w", err)
	}

	return todo, nil
}

// Create creates a new todo item.
func (r *TodoRepository) Create(req models.TodoRequest) (*models.Todo, error) {
	todo := &models.Todo{}
	query := `
        INSERT INTO todos (title, description, completed) 
        VALUES ($1, $2, $3) 
        RETURNING id, title, description, completed, created_at, updated_at`

	err := r.db.Get(todo, query, req.Title, req.Description, req.Completed)
	if err != nil {
		return nil, fmt.Errorf("failed to create todo: %w", err)
	}

	return todo, nil
}

// Update updates an existing todo item.
func (r *TodoRepository) Update(id int, req models.TodoRequest) (*models.Todo, error) {
	todo := &models.Todo{}
	query := `
        UPDATE todos 
        SET title = $1, description = $2, completed = $3, updated_at = NOW() 
        WHERE id = $4 
        RETURNING id, title, description, completed, created_at, updated_at`

	err := r.db.Get(todo, query, req.Title, req.Description, req.Completed, id)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("todo with id %d not found", id)
		}
		return nil, fmt.Errorf("failed to update todo: %w", err)
	}

	return todo, nil
}

// Delete removes a todo item by its ID.
func (r *TodoRepository) Delete(id int) error {
	query := "DELETE FROM todos WHERE id = $1"
	result, err := r.db.Exec(query, id)
	if err != nil {
		return fmt.Errorf("failed to delete todo: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get affected rows: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("todo with id %d not found", id)
	}

	return nil
}
