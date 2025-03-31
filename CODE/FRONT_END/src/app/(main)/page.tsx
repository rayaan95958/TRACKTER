import Container from "../../components/container/Container";
import Contact from "../../components/contact/Contacts"
import YellowButton from "../../components/button/YellowButton";
import GreyButton from "../../components/button/GreyButton";

export default function Home() {
  return (
    <>
      <div className="flex justify-center mt-10 md:mt-12 lg:mt-16 gap-4 lg:gap-8">
        <YellowButton text="Main Page" href="/" />
        <GreyButton text="Database" href="/database" />
      </div>
      <div className="w-full bg-black flex flex-col items-center justify-center overflow-hidden rounded-md">
        <div className="w-full h-full relative">
          <Container />
          <Contact />
        </div>
      </div>
    </>
  );
}
