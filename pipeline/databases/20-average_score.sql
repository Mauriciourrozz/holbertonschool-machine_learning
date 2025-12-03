-- Stored procedure ComputeAverageScoreForUser: computes and stores the average score of a user
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    -- Calcular el promedio de las correcciones del usuario
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE corrections.user_id = user_id;

    -- Guardar el promedio en la tabla users
    UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;

END$$

DELIMITER ;
