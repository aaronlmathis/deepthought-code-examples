package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

type DB struct {
	*sqlx.DB
}

// NewConnection creates a new database connection
func NewConnection(config Config) (*DB, error) {
	connectionString := config.ConnectionString()

	sqlxDB, err := sqlx.Connect("postgres", connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	sqlxDB.SetMaxOpenConns(config.MaxOpenConns)
	sqlxDB.SetMaxIdleConns(config.MaxIdleConns)
	sqlxDB.SetConnMaxLifetime(config.ConnMaxLifetime)
	sqlxDB.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// Verify connection
	if err := sqlxDB.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DB{sqlxDB}, nil
}

// Close closes the database connection
func (db *DB) Close() error {
	return db.DB.Close()
}

// HealthCheck verifies database connectivity
func (db *DB) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var result int
	err := db.GetContext(ctx, &result, "SELECT 1")
	if err != nil {
		return fmt.Errorf("database health check failed: %w", err)
	}

	return nil
}

// GetStats returns database connection pool statistics
func (db *DB) GetStats() sql.DBStats {
	return db.DB.Stats()
}
