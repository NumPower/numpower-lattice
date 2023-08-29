<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Core\Variable;

interface ILayer
{
    function __invoke(Variable $inputs): Variable;
    function build(array $inputShape);
    function generateOutputShape(): array;
    function getInputShape(): array;
    function getName(): string;
    function built(): bool;
    function countParams(): int;
    function isTrainable(): bool;
    function countTrainableParams(): int;
    function countNonTrainableParams(): int;

    /**
     * @return Variable[]
     */
    function getTrainableWeights(): array;
}