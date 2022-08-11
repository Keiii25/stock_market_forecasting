const fs = require("fs");

const imageFileNames = () => {
  const array = fs
    .readdirSync("assets/images")
    .filter((file) => {
      return file.endsWith(".png");
    })
    .map((file) => {
      return file
        .replace("@2x.png", "")
        .replace("@3x.png", "")
        .replace(".png", "");
    });

  return Array.from(new Set(array));
};

const generate = () => {
  let properties = imageFileNames()
    .map((name) => {
      return `${name}: require('./images/${name}.png')`;
    })
    .join(",\n  ");

  const comment =
    "//Run command ' node scripts/generate-images.js ' at root folder to auto regenerate this file when adding / removing images from app/assets/image folder  \n  \n ";
  const string =
    comment +
    `const images = {
  ${properties}
}

export default images
`;

  fs.writeFileSync("assets/images.js", string, "utf8");
};

generate();
