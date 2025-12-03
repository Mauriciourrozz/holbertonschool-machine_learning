-- Function SafeDiv, safely divides a by b, returns 0 if b is 0
DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT DETERMINISTIC
BEGIN
    -- Si el divisor es 0, devolver 0
    IF b = 0 THEN
        RETURN 0;
    END IF;

    -- Si no, devolver a / b
    RETURN a / b;
END$$

DELIMITER ;
