package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/jmoiron/sqlx"
)

// QueryTimer wraps database operations with timing metrics
type QueryTimer struct {
	db *sqlx.DB
}

func NewQueryTimer(db *sqlx.DB) *QueryTimer {
	return &QueryTimer{db: db}
}

func (qt *QueryTimer) SelectContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	start := time.Now()
	err := qt.db.SelectContext(ctx, dest, query, args...)
	duration := time.Since(start)

	logQuery("SELECT", query, duration, err)
	return err
}

func (qt *QueryTimer) GetContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	start := time.Now()
	err := qt.db.GetContext(ctx, dest, query, args...)
	duration := time.Since(start)

	logQuery("GET", query, duration, err)
	return err
}

func (qt *QueryTimer) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	start := time.Now()
	result, err := qt.db.ExecContext(ctx, query, args...)
	duration := time.Since(start)

	logQuery("EXEC", query, duration, err)
	return result, err
}

func logQuery(operation, query string, duration time.Duration, err error) {
	status := "SUCCESS"
	if err != nil {
		status = "ERROR"
	}

	// In production, use structured logging
	fmt.Printf("[DB] %s %s - Duration: %v - Status: %s\n",
		operation, shortenQuery(query), duration, status)

	// Log slow queries (> 100ms)
	if duration > 100*time.Millisecond {
		fmt.Printf("[SLOW QUERY] %s - Duration: %v\n", shortenQuery(query), duration)
	}
}

func shortenQuery(query string) string {
	if len(query) > 100 {
		return query[:100] + "..."
	}
	return query
}
