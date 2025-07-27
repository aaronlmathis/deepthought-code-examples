package database

import (
	"fmt"

	"github.com/jmoiron/sqlx"
)

// TxWrapper wraps sqlx.Tx to implement our Tx interface
type TxWrapper struct {
	*sqlx.Tx
}

// Beginx starts a new transaction
func (db *DB) Beginx() (Tx, error) {
	tx, err := db.DB.Beginx()
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	return &TxWrapper{Tx: tx}, nil
}

// Commit commits the transaction
func (tx *TxWrapper) Commit() error {
	if err := tx.Tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	return nil
}

// Rollback rolls back the transaction
func (tx *TxWrapper) Rollback() error {
	if err := tx.Tx.Rollback(); err != nil {
		return fmt.Errorf("failed to rollback transaction: %w", err)
	}
	return nil
}
