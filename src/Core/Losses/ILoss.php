<?php

namespace NumPower\Lattice\Core\Losses;

interface ILoss
{
    function __invoke(\NDArray $true, \NDArray $pred): float;
}