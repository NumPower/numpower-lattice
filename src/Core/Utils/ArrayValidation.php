<?php

namespace NumPower\Lattice\Core\Utils;

use NumPower\Lattice\Exceptions\ValueErrorException;

class ArrayValidation
{
    /**
     * @param string $argname
     * @param array $values
     * @return void
     * @throws ValueErrorException
     */
    public static function intArrayOrFail(string $argname, array $values): void
    {
        if (count($values) == 0) {
            return;
        }
        foreach ($values as $val) {
            if (!is_int($val)) {
                throw new ValueErrorException(
                    "Argument `$argname` must be a array with integer values. Received: $argname=" . gettype($val)
                );
            }
        }
    }
}