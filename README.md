# Abstract Algebra Visualizer

Just a simple tool for visualizing groups and eventually other abstract objects written with the Rust programming language and OpenGL.

## Installation

Either download and extract the zip manually or use git to clone the repository which I recommend for easy updates.


To clone use:
```bash
git clone https://github.com/DamonPalovaara/Abstract-Algebra-Visualizer
```

To update the repository use:
```bash
git pull
```

For more instructions on how to install and use git follow this [link](rogerdudler.github.io/git-guide/)

## Running

Before you can run the program you need to have Rust installed on your computer.

Instructions for that can be found [here](www.rust-lang.org/tools/install)

Once Rust has been installed you can use cargo (Rust's package manager) to compile and run the program by typing the following command inside the working directory.
```bash
cargo run
```

If you're having any issue getting the program to run feel free to email me at dpalovaa@nmu.edu

## Known Issues

You can only do one flip before having to reset. The issue has to do with the way I've implemented the rotational matrices. It's an easy fix but I'm catching up on some sleep first.

## Future features

I want to make a main menu that will let you select between different groups to visualize.

Soon I will implement text so the buttons and the current state have labels.

Other groups I want to visualize are the symmetries of 3D shapes, direct product groups, relatively prime groups, etc.

If you have any suggestions on features to add feel free to send me an email!

## Contributing
Pull requests are welcomed. For major changes, please open an issue first to discuss what you would like to change.

## License
TODO: I want this project to be open source