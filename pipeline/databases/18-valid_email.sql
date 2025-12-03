-- Trigger that resets valid_email only when the email has changed
DELIMITER $$

CREATE TRIGGER reset_email_validity
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    -- Si el email nuevo es distinto del anterior, se resetea valid_email
    IF NEW.email <> OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END$$

DELIMITER ;
