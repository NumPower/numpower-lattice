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
    public function __invoke(Variable $inputs, bool $training = false): Variable;

    /**
     * @param int[] $inputShape
     * @return void
     */
    public function build(array $inputShape): void;

    /**
     * @return int[]
     */
    public function generateOutputShape(): array;

    /**
     * @return int[]
     */
    public function getInputShape(): array;

    /**
     * @return string
     */
    public function getName(): string;

    /**
     * @return bool
     */
    public function built(): bool;

    /**
     * @return int
     */
    public function countParams(): int;

    /**
     * @return bool
     */
    public function isTrainable(): bool;

    /**
     * @return int
     */
    public function countTrainableParams(): int;

    /**
     * @return int
     */
    public function countNonTrainableParams(): int;

    /**
     * @return Variable[]
     */
    public function getTrainableWeights(): array;
}
