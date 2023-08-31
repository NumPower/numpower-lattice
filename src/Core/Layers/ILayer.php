<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Core\Variable;

interface ILayer
{
    /**
     * @param Variable $inputs
     * @param bool $training
     * @return Variable
     */
    function __invoke(Variable $inputs, bool $training = False): Variable;

    /**
     * @param array $inputShape
     * @return mixed
     */
    function build(array $inputShape);

    /**
     * @return array
     */
    function generateOutputShape(): array;

    /**
     * @return array
     */
    function getInputShape(): array;

    /**
     * @return string
     */
    function getName(): string;

    /**
     * @return bool
     */
    function built(): bool;

    /**
     * @return int
     */
    function countParams(): int;

    /**
     * @return bool
     */
    function isTrainable(): bool;

    /**
     * @return int
     */
    function countTrainableParams(): int;

    /**
     * @return int
     */
    function countNonTrainableParams(): int;

    /**
     * @return Variable[]
     */
    function getTrainableWeights(): array;
}