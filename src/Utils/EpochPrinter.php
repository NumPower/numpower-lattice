<?php

namespace NumPower\Lattice\Utils;

class EpochPrinter
{

    private int $rotatingIndex = 0;

    private int $startTime = 0;

    /**
     * @param int $epoch
     * @param int $total_epochs
     * @param string $device
     * @return void
     */
    public function start(int $epoch, int $total_epochs, string $device) {
        print("\e[107m\e[30m Epoch ". $epoch + 1 . "/$total_epochs [".$device."] \e[0m\n");
        $this->startTime = time();
    }

    /**
     * @param int $current
     * @param int $total
     * @return void
     */
    public function update(int $current, int $total): void {
        $string = "";
        $rotatingCharacters = ["\u{25DC}", "\u{25DD}", "\u{25DE}", "\u{25DF}"];
        $totalSteps = $total;
        $step = $current;

        $progress = ($step / $totalSteps) * 100;
        $string .= "\r";
        $string .= "Progress: ";
        $string .= $step . "/" . $totalSteps;
        $string .= " [";
        $barLength = 20; // Adjust the length of the progress bar
        $barFill = round(($progress / 100) * $barLength);
        $string .= str_repeat("\u{25A0}", $barFill);
        if ($progress < 100) {
            $string .= "" . $rotatingCharacters[$this->rotatingIndex];
        }
        $this->rotatingIndex = ($this->rotatingIndex + 1) % count($rotatingCharacters);
        $string .= str_repeat(' ', $barLength - $barFill);
        $string .= "] " . round($progress) . "%";


        $elapsedTime = time() - $this->startTime;
        $eta = (int)(($elapsedTime / $step) * ($totalSteps - $step));
        $string .= " Elapsed: " . gmdate("H:i:s", $elapsedTime);
        $string .= " ETA: " . gmdate("H:i:s", $eta);
        print $string;
    }

    /**
     * @return void
     */
    public function stop() {
        echo "\n";
    }
}